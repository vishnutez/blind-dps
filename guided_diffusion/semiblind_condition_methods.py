from typing import Dict
import torch

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator
from guided_diffusion.blind_condition_methods import BlindConditioningMethod

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class SemiblindConditioningMethod(BlindConditioningMethod):
    def __init__(self, operator, noiser=None, **kwargs):
        '''
        Handle multiple score models.
        Yet, support only gaussian noise measurement.
        '''
        assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator)
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def grad_and_value(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):

        if self.noiser.__name__ == 'gaussian' or self.noiser is None:  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]

            # print('x_0_hat_values:', x_0_hat_values)
            
            difference = measurement - self.operator.forward(*x_0_hat_values)
            blind_norm = torch.linalg.norm(difference)

            # Additional guidance from ys, xs

            ys = kwargs.get('ys', None)
            xs = kwargs.get('xs', None)

            if xs is not None and ys is not None:
                print('Driving the diffusion with additional guidance from the guidance samples.')
                sample_difference = ys - self.operator.forward(xs, x_0_hat['kernel'])
                norm = blind_norm + torch.linalg.norm(sample_difference)
                print('x_0_hat_values[1]:', x_0_hat_values[1].shape)
            else:
                print('No ys, xs, just using the blind DPS.')
                norm = blind_norm

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        norm = norm + reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord)                        
                    
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
            
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm

@register_conditioning_method(name='ps')
class PosteriorSampling(SemiblindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev, x_0_hat, measurement, **kwargs)

        scale = kwargs.get('scale')
        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())
        for k in keys:
            x_t.update({k: x_t[k] - scale[k]*norm_grad[k]})            
        
        return x_t, norm
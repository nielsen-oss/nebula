"""Module for meta-classes."""

import inspect
from copy import deepcopy


class InitParamsStorage(type):

    def __call__(cls, *args, **kwargs):
        """Store the initialization parameters."""
        try:
            obj = type.__call__(cls, *args, **kwargs)
        except TypeError as e:
            err = str(e.with_traceback(None))
            if "got an unexpected keyword argument" not in err:
                raise e

            # print the __init__ signature to help debugging
            sign = inspect.signature(cls.__init__)
            params = sign.parameters
            li_kwargs = [str(v) for k, v in params.items() if k != "self"]
            sign_str = "\n  ".join(li_kwargs)
            err += f"\n__init__ signature: \n  {sign_str}"
            raise TypeError(err)

        obj._transformer_init_params = kwargs
        return obj


if __name__ == "__main__":  # pragma: no cover
    # Print initialization parameters stored with InitParamsStorage
    class Parent(metaclass=InitParamsStorage):
        def __init__(self):
            """Emulate the base Transformer class."""
            self._transformer_init_params: dict = {}

        @property
        def transformer_init_parameters(self) -> dict:
            """Return the initialization parameters."""
            return deepcopy(self._transformer_init_params)


    class Child(Parent):  # pragma: no cover
        def __init__(self, *, my_param):
            """Emulate a generic Transformer class."""
            super().__init__()
            self._my_param = my_param


    a = Child(my_param="my_value")
    print(a.transformer_init_parameters)

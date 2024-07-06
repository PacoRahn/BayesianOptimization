class Kernel_Factory:
    kernel_reg = {}

    @staticmethod
    def register_kernel(name):
        def decorator(cls):
            Kernel_Factory.kernel_reg[name] = cls
            return cls
        return decorator

    @staticmethod
    def create_kernel(kernel_type,**kwargs):
        if kernel_type not in Kernel_Factory.kernel_reg:
            raise ValueError(f"Invalid Kernel name: {kernel_type} - is it imported correctly? ")
        return Kernel_Factory.kernel_reg[kernel_type](**kwargs)

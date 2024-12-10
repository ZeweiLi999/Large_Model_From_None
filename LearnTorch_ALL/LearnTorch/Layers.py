from LearnTorch import Parameter

class Layer:
    def __init__(self):
        self._params = set() # 保存了Layer的所有参数
        # set类元素没有顺序，且不会有ID相同的对象

    def __setattr__(self, name, value):# __setattr__方法是在设置实例变量时被调用的特殊方法
        # 参数：实例变量的名字作为name传入、实例变量的值作为value传入
        if isinstance(value, Parameter): # 如果是Parameter类，就添加名字到参数
            self._params.add(name) # 这么做是因为将参数保存到外部文件时，保存name会更方便
        super().__setattr__(name, value) # 继续通过 super() 调用父类的 __setattr__ 方法来完成正常的属性设置。
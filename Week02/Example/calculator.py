class SetData:
    def __init__(self):
        self.x = 0
        self.y = 0
        print("부모 클래스 초기화")

    def SetX(self, x):
        self.x = x

    def SetY(self, y):
        self.y = y

    def SetXandY(self, x, y):
        self.x = x
        self.y = y


class Mycalculator(SetData):
    def __init__(self):
        super(Mycalculator, self).__init__()
        print("자식 클래스 초기화")

    def Add(self):
        res = self.x + self.y
        return res

    def Subtract(self):
        res = self.x - self.y
        return res


cal = Mycalculator()

cal.SetXandY(10, 20)
cal.SetX(30)

res = cal.Add()

print(res)




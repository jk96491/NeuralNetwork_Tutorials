
class calculator:
    def __init__(self):
        print("여기서 초기화")
        self.x = 0
        self.y = 0

    def SetX(self, x):
        self.x = x

cal = calculator()

cal.SetX(10)

print(cal.x)




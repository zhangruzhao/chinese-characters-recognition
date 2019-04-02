class Stu(object):
    def __init__(self):
        self.name_list = []
        self.name = ""
        self.printf()

    def printf(self):
        for i in range(5):
            self.name = "zhang"
            self.name_list.append(self.name)
        print(self.name_list)

if __name__ == '__main__':
    stu = Stu()

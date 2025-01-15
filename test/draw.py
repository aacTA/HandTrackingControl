import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def line_generator(filepath):
    with open(filepath,'r')as file:
        for line in file:
            ## your code
            yield line

lines = line_generator('point.txt')

x = []
y = []
z = []
i = 0
count = 0
flag = 1
for line in lines:
    if i % 2 == 0:
        if line.strip() == 'True':
            flag = 1
        else:
            flag = 0
    else:
        if flag == 1:
            x.append(count)
            count = count + 1

            y.append(float(line))
            print(float(line))
    i += 1

# 创建图形
plt.plot(x, y)
# 显示图形
plt.savefig('plt.png')



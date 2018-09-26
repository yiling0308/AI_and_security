
#  建立線性迴歸分析模型:用氣溫來預測冰紅茶的銷售量

https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day21.md

使用 sklearn.linear_model 的 LinearRegression() 方法

```
import numpy as np
from sklearn.linear_model import LinearRegression

# 資料==> 使用np.array來建立氣溫及冰紅茶的銷售量
temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

# 先建立線性迴歸的物件
lm = LinearRegression()

# 再用線性迴歸物件的fit()函式進行分析
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))


# 印出係數
print(lm.coef_)

# 印出截距
print(lm.intercept_ )

```
執行
```
python2 abc.py
  File "abc.py", line 4
SyntaxError: Non-ASCII character '\xe8' in file abc.py on line 4, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details
```
錯誤更正==>加入頭兩欄

```
#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import numpy as np
from sklearn.linear_model import LinearRegression

# 資料==> 使用np.array來建立氣溫及冰紅茶的銷售量
temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

# 先建立線性迴歸的物件
lm = LinearRegression()

# 再用線性迴歸物件的fit()函式進行分析
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))


# 印出係數
print(lm.coef_)

# 印出截距
print(lm.intercept_ )

```
再次執行
```
python2 abc.py
[[ 3.73788546]]
[-36.36123348]

```


###  使用線性迴歸分析模型預測:用氣溫來預測冰紅茶的銷售量

https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day21.md

使用 LinearRegression() 的 predict() 方法
```
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 新的氣溫
to_be_predicted = np.array([30])
predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))

# 預測的冰紅茶銷量
print(predicted_sales)
```

### 線性迴歸視覺化:使用 matplotlib.pyplot 的 scatter() 與 plot() 方法
```
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 新的氣溫
to_be_predicted = np.array([30])
predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))

# 視覺化
plt.scatter(temperatures, iced_tea_sales, color='black')
plt.plot(temperatures, lm.predict(np.reshape(temperatures, (len(temperatures), 1))), color='blue', linewidth=3)
plt.plot(to_be_predicted, predicted_sales, color = 'red', marker = '^', markersize = 10)
plt.xticks(())
plt.yticks(())
plt.show()

```

### 線性迴歸模型的績效:使用線性迴歸模型的績效（Performance|**Mean squared error（MSE）**與 R-squared
```
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

# 轉換維度
temperatures = np.reshape(temperatures, (len(temperatures), 1))
iced_tea_sales = np.reshape(iced_tea_sales, (len(iced_tea_sales), 1))

lm = LinearRegression()
lm.fit(temperatures, iced_tea_sales)

# 模型績效
mse = np.mean((lm.predict(temperatures) - iced_tea_sales) ** 2)
r_squared = lm.score(temperatures, iced_tea_sales)

# 印出模型績效
print(mse)
print(r_squared)

```

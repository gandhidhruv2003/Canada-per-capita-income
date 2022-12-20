import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")

# Training the dataset
reg = linear_model.LinearRegression()
reg.fit(df[["year"]], df.price)

plt.xlabel("Year")
plt.ylabel("Per capita income (US$)")
plt.scatter(df.year, df.price, color="Red")  # Putting the points
plt.plot(df.year, reg.predict(df[["year"]]), color="Blue")  # Ploting the best fit line
plt.show()
print(reg.score([2022]))
print("Per capita income (US$) in 2020 is", reg.predict([[2020]])[0])
print("Per capita income (US$) in 3300 is", reg.predict([[3300]])[0])

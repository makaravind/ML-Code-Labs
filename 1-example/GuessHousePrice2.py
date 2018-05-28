import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from GuessHousePrice import getPreprocessedData, getTest

df, features, target = getPreprocessedData()

x_train, y_train, x_test, y_test = df[features][:100],df[target][:100], df[features][100: 200], df[target][100: 200]

# y = mx + c --> y_hat = w1 * x + w0 --> y_hat = w1 * x1 + w2 * x2 + ... w5 * x5 + w0
model = LinearRegression()
model.fit(x_train, y_train)

# x_test = getTest()
y_hat = model.predict(x_test.head(10))
print( 'y here', y_hat)
print('y original here', y_test.head(10))

# y = w0 + w1x + w2x2 + ...
print(' coefficients ', model.coef_)

# Plot outputs
plt.scatter(x_test.LotArea.head(10), y_test.head(10),  color='black')
plt.plot(x_test.LotArea.head(10), y_hat, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

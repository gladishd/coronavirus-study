# approximate_index.py
# Welcome - This file is about finding different ways to estimate the .DJI with less variables!
# Please let me know if something is broken here because I want to fix it sooner rather than later.
# Thank you and enjoy your stay.
import numpy as np  # to edit data frames
import math  # math functions
import matplotlib.pyplot as plt  # to visualize
# to calculate pearson correlation coefficient and p-value of model
from scipy import stats
import pandas as pd  # To read .csv and mutate data frames
# to create the multilinear regression model
from sklearn.linear_model import LinearRegression
import sys  # to access command line args
# to convert dates (2019-03-20, ..., 2019-09-27) into integers (0, ..., 191).
from datetime import date
from textwrap import wrap  # so we can keep large equations on the graph

data = pd.read_csv(sys.argv[3])  # read in the data
# turn it into a pandas dataframe
df = pd.DataFrame(data, columns=['Symbol', 'Date', 'Close'])

# Want: a separate column for each symbol.
newColumns = np.append(['DATE'], data.Symbol.unique())
# So newColumns will be [DATE, symbol1, symbol2, ..., symbol30]

# Define a function that takes the current day string, such as 2019-05-25, and
# returns the distance in days to 2019-03-20, the starting point.


def convertDay(day):
    arr1 = "2019-03-20".split("-")
    arr2 = day.split("-")
    arr1 = list(map(lambda x: int(x), arr1))
    arr2 = list(map(lambda x: int(x), arr2))
    first, second, third = arr1
    f_date = date(first, second, third)
    first, second, third = arr2
    l_date = date(first, second, third)
    delta = l_date - f_date
    return (delta.days)


# Based on the original Date column, make a new Date column with integer representations of each day
newDates = list(map(lambda x: convertDay(x), data['Date']))
df['Date'] = newDates  # Update the Date column

# create dfObj, an empty dataframe with the following column names:
# DATE, .DJI, AAPL, AXP, BA, CAT, CSCO, CVX, DIS, DOW,
# GS, HD, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT,
# NKE, PFE, PG, TRV, UNH, UTX, V, VZ, WBA, WMT, XOM
dfObj = pd.DataFrame(columns=newColumns)  # Now, let's start populating dfObj:

for day in range(0, 192):  # for all days 0, 1, ..., 191
    aa = df[df.Date == day]  # find all rows matching the current day
    values = aa["Close"].to_numpy()  # Get all closing prices for that day
    # print('values is ', values, len(values))
    if (values.any()):  # (if we had data for that day)
        dfObj.loc[day] = [day, values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14],
                          values[15], values[16], values[17], values[18], values[19], values[20], values[21], values[22], values[23], values[24], values[25], values[26], values[27], values[28], values[29], values[30]]

# print(dfObj)
# print(data)
# # The translation was successful!

# from the command line args supplied by the user
indexY = list(dfObj.columns).index(sys.argv[2])

# Get all columns.  Each symbol is one column, showing closing prices from the start to end date.
X = dfObj.iloc[:, 1:32]
X = X.drop([sys.argv[2]], axis=1)
X = X.values.reshape(-1, 30)
# (numpy should calculate the dimension of rows, and assume there are 30 columns.  One for each symbol)

# Get the .DJI column
Y = dfObj.iloc[:, indexY].values.reshape(-1, 1)
# (numpy should calculate the dimension of rows, with 1 column for Y)

# this is just a preliminary multilinear model with all 30 symbols
model = LinearRegression(fit_intercept=False).fit(X, Y)

Y = Y.flatten()  # remove the list encasing number
# and make it comma separated, because we're about to calculate the pearson coefficient of correlation
Y = list(Y)

modelCoefficients = list(model.coef_.flatten())
# an array of all coefficients!  We have to rank these based on greatest pearson coefficient of correlation,
# then statistically significant p-value (as long as it's less than 0.05 then the symbol is statistically significant).
# I am not using the coefficients from the all-30-symbols model because they are not standardized and assume a strong linear correlation with .DJI.

stockRankings = [None] * 30
fig, ax = plt.subplots(5, 6)  # This is for the visualization
fig.suptitle('All Symbols against the ' + sys.argv[2])
symbolsOriginal = list(data.Symbol.unique())
symbolsOriginal.pop(symbolsOriginal.index(sys.argv[2]))

for stockID in range(0, 30):  # For each stock, find its importance
    prices130DaysForStock = [None] * 130
    for i in range(0, 130):
        prices130DaysForStock[i] = X[i][stockID]
    pearson_coef, p_value = stats.pearsonr(prices130DaysForStock, Y)
    stockRankings[stockID] = [pearson_coef, p_value,
                              modelCoefficients[stockID], stockID, symbolsOriginal[stockID]]
    # So each entry in stockRankings is
    # [Pearson Coefficient, P-Value, Coefficient in all-30 linear model, its index out of all stocks in alphabetical order, and the symbol itself]

    # The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect)
    # Want: lowest p-value, greatest pearson coefficient, greatest coefficient
    ax[math.floor(stockID / 6), stockID %
       6].scatter(prices130DaysForStock, Y, s=5, alpha=0.5, c='b')
    # Plot the closing price of the symbol, and the closing price of the Dow Jones Industrial Average

    ax[math.floor(stockID / 6), stockID %
       6].set(title=symbolsOriginal[stockID])  # set title
    ax[math.floor(stockID / 6), stockID % 6].tick_params(axis='both',
                                                         labelsize=3, length=0)  # remove ticks

for a in ax.flat:
    a.label_outer()  # only show outer labels and tick labels

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # set spacing between subplots

plt.show()  # render the figure

# sort from largest pearson correlation coefficient to smallest
stockRankings = sorted(stockRankings, key=lambda x: (-x[0]))

# If the p-value is greater than 0.05, then there's a chance that the symbol isn't really correlated with the .DJI index
# So, we need to move symbols with large p-values (low statistical significance) to the end of the ranking
statisticallySignificantValues = list(
    filter(lambda x: x[1] <= 0.05, stockRankings))

# Anything with a p-value larger than 5% means the stock may really not be correlated with the .DJI at all.
# That's the tradition in statistics, so I'm just using 0.05 as the cutoff point.
nonSignificantValues = filter(lambda x: x[1] > 0.05, stockRankings)

# For these non-significant values, let's sort them from largest coefficient to smallest coefficient
# I want the variables to have the greatest influence on the Y value.
nonSignificantValues = sorted(nonSignificantValues, key=lambda x: (-x[2]))

# Re-combine the two sets
stockRankings = statisticallySignificantValues + nonSignificantValues

# Take the first n symbols, those which have the highest ranking
stockRankingsTopN = stockRankings[:int(sys.argv[1])]

# This will have the indices of the symbols we are going to include
columnIndices = []

# Find which column numbers from dfObj correspond to the top N stocks we found
for x in stockRankingsTopN:
    columnIndices.append(list(dfObj.columns).index(x[4]))
columnIndices = list(columnIndices)

# From the existing dfObj, take only the columns representing the N most important variables
dfObjTopNSymbols = dfObj.iloc[:, columnIndices]

# Convert the dataframe back into an array.
# Within this array, each subarray represents one row of the original dataframe.
dfObjTopNSymbols = dfObjTopNSymbols.values.reshape(-1, int(sys.argv[1]))

# So Y is an array with all .DJI index values
model = LinearRegression(fit_intercept=False).fit(dfObjTopNSymbols, Y)

# The output of the model needs to be flattened and converted into the list datatype
modelCoefficients = list(model.coef_.flatten())

# Want to print this or alternatively write this to .csv depending on how approximate_index.py is invoked
# (python3 approximate_index.py 3 .DJI dow_jones_historical_prices.csv > approximation.csv)
res = pd.DataFrame(columns=['Symbol', 'Weight'])
# Now, find the columns at those indices and then get the column names from the .columns property
res['Symbol'] = list(dfObj.iloc[:, columnIndices].columns)
# Round it to 3 places after the decimal
res['Weight'] = np.around(modelCoefficients, 3)
print(res)  # This is the only system output we want, because we are writing to .csv

# Predict the response
y_pred = model.intercept_ + np.sum(model.coef_*dfObjTopNSymbols, axis=1)
fig, ax = plt.subplots()
ax.scatter(y_pred, Y, s=5, alpha=0.5, c='b')

# String together the title
title = sys.argv[2] + '= '
for i in range(len(res['Weight'])):
    title += str(int(res['Weight'][i])) + res['Symbol'][i] + ' + '

# Remove the extra + sign
title = title[:len(title) - 3]

# Add the coefficient of determination (R^2) just for fun
title = title + ', R^2 = ' + \
    str(np.around(model.score(dfObjTopNSymbols, Y), 3))

plt.title('\n'.join(wrap(title, 60)), fontsize=10)
plt.xlabel('Predicted Value of ' + sys.argv[2])
plt.ylabel('True Value of ' + sys.argv[2])
plt.show()

## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df

~~~

![image](https://github.com/user-attachments/assets/ef8d1fb8-c929-49ee-ba09-2dae5940c413)

~~~

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

~~~

![image](https://github.com/user-attachments/assets/e3112622-2f4c-4f1d-89bb-1561692e6fe5)

~~~

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

~~~

![image](https://github.com/user-attachments/assets/ad8ec125-f4ea-43e5-a83e-81168f01dd50)

~~~

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

~~~

![image](https://github.com/user-attachments/assets/3dc199b7-0fbe-4836-a04b-cd91f84f5ac2)


~~~

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

~~~

![image](https://github.com/user-attachments/assets/bbcf0982-1dda-4385-9b73-611bcb6a6dde)

~~~

pd.get_dummies(df2,columns=["nom_0"])

~~~

![image](https://github.com/user-attachments/assets/b9877591-c345-4fa9-b23d-9a13d4a61e98)

~~~

pip install --upgrade category_encoders

~~~

![image](https://github.com/user-attachments/assets/8c79a66c-25f2-4c26-b117-7039340ac2e5)

~~~

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df

~~~

![image](https://github.com/user-attachments/assets/a8aa4527-a9c5-4648-ba95-10b3d80b8c31)

~~~

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

~~~

![image](https://github.com/user-attachments/assets/87b0604b-038c-4422-893d-b629ddd56341)

~~~

dfb=pd.concat([df,nd],axis=1)
dfb

~~~

![image](https://github.com/user-attachments/assets/f66459a9-9003-4f7e-a554-e015e875f337)

~~~

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

~~~

![image](https://github.com/user-attachments/assets/72a93019-e464-496d-a6a1-3608d8faf083)

~~~

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

~~~

![image](https://github.com/user-attachments/assets/c0299480-f5fc-483d-8233-f9e927f89f29)

~~~

df.skew()

~~~

![image](https://github.com/user-attachments/assets/fdbbd494-8629-48c4-8468-4fa90b145696)

~~~

np.log(df["Highly Positive Skew"])

~~~

![image](https://github.com/user-attachments/assets/210f0824-394c-4985-934e-668170989b85)

~~~

np.reciprocal(df["Moderate Positive Skew"])

~~~

![image](https://github.com/user-attachments/assets/5e2bf1ec-cd73-4a6a-baba-6194159ce4c0)

~~~

np.sqrt(df["Highly Positive Skew"])

~~~

![image](https://github.com/user-attachments/assets/5f77a6ed-905d-4d73-8c3d-b85db14daa2c)

~~~

np.square(df["Highly Positive Skew"])

~~~

![image](https://github.com/user-attachments/assets/924d4ef2-695c-45c2-ae0b-b00dd75db1e8)

~~~

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

~~~

![image](https://github.com/user-attachments/assets/c79a286e-2277-49ba-84a4-2bee8a1a375e)

~~~

df.skew()

~~~

![image](https://github.com/user-attachments/assets/c263074d-5750-4a37-9f02-b7bf01bf00d1)

~~~

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

~~~

![image](https://github.com/user-attachments/assets/8ccdcfc6-b168-45d9-b5bf-8aa1f01b1414)

~~~

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

~~~

![image](https://github.com/user-attachments/assets/418a31b1-0ed2-4f4a-a5ba-7885897cdfd0)

~~~

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

~~~

![image](https://github.com/user-attachments/assets/4039b646-086b-4c8b-af49-4b224a737ffb)

~~~

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

~~~

![image](https://github.com/user-attachments/assets/ad9fde04-e3e0-4cfb-85b8-7d4dbcfdf77e)

~~~

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

~~~

![image](https://github.com/user-attachments/assets/89c72d39-608a-4956-bc0a-0f0ddee50270)

~~~

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

~~~

![image](https://github.com/user-attachments/assets/a0913a1d-8793-47a8-bc5c-ae4697515c1b)

~~~

dt=pd.read_csv("/content/titanic_dataset.csv")
dt

~~~

![image](https://github.com/user-attachments/assets/0c6ef87d-28c5-4d73-bf07-5864036d49b9)

~~~

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

~~~

![image](https://github.com/user-attachments/assets/2885aa9b-5baa-4368-a138-bee585d49441)

~~~

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

~~~

![image](https://github.com/user-attachments/assets/baf66553-6be7-47d1-8e26-5e9136d36b24)


# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully
       

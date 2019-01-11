#Tools for DataSet manupulation
import pandas as pd
from collections import Counter
import numpy as np
import math
from scipy import stats
import random

#Pre Processing using sklearn
from sklearn import preprocessing

#Validation using sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score

#Regression Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, LassoCV, ElasticNetCV, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#plots
import matplotlib.pyplot as plt

def readTheData(fname):
    return pd.read_csv(fname)

def printSkewness(df):
    for c in list(df.select_dtypes(include=[np.number])):
        if df[c].skew() < -1 or df[c].skew() > 1:
            print("Skewness of",c,"is : ",df[c].skew()," and kurtosis : ",df[c].kurtosis())

def printCorr (df, lowerLimit = .7):
    corr_df = df.corr()

    for c in list(corr_df):
        temp_df = corr_df[(corr_df[c].abs() > .7) & (corr_df[c].abs() < 1) ]
        if not temp_df.empty:
            print(temp_df[c], "\n\n")


def removeDuplicate(df):
    
    print('Initial df size', df.shape)
    
    print('Duplicates size', df[df.duplicated()].shape)
    
    print('NaN columns :', df.columns[df.isna().any()].tolist())
    
    df = df.fillna('NA')
    print('NaN columns (post fillna):', df.columns[df.isna().any()].tolist())
    print('Duplicates size (post fillna)', df[df.duplicated()].shape)
    
       
    key_columns = [x for x in list(df) if x != 'time']
    print("Duplicates without considering *time* feature : ", df[df.duplicated(key_columns)].shape)
    df = df.sort_values(by='time', ascending=False).reset_index(drop=True)
    
    df = df.drop_duplicates(key_columns,keep='first').reset_index(drop=True)
    df = df.drop_duplicates()
    
    print("Post duplicates removal data set size : ", df.shape)
    
    return df
    
def removeOutlier(df):
    print("\nShape of the data set before to outlier removal : ", df.shape)
    df = df[df.Census_MedianIncome != 0]
    print("Shape of the data set after outlier removal : ", df.shape)
    print()
    
    return df
    
def dropCorrelatedFeatures(df):
    print("\nShape of the data set before dropping co-related features : ", df.shape)
    df = df.drop(['CollegeGrads','physician_dist_miles','opt_dist_miles'],axis=1)
    print("Shape of the data set post dropping co-related features : ", df.shape)
    print()
    return df

def getDummiesForBooleanFeatures(df):
    print("\nShape of the data set before creating dummies : ", df.shape)
    
    df.garage = df.garage.apply(lambda x: 'garage_yes' if x == 0 else 'garage_no' )
    df.pool = df.pool.apply(lambda x: 'pool_yes' if x == 0 else 'pool_no' )
    df.fireplace = df.fireplace.apply(lambda x: 'fireplace_yes' if x == 0 else 'fireplace_no' )
    df.patio = df.patio.apply(lambda x: 'patio_yes' if x == 0 else 'patio_no' )
    
    categorical_columns = ['garage', 'pool', 'fireplace', 'patio']
    for c in categorical_columns:
        temp = pd.get_dummies(df[c])
        df[temp.columns] = temp
    df = df.drop(categorical_columns, axis=1)
    
    print("Shape of the data set post creating dummies : ", df.shape)
    print()
    return df


def preProcessTheData(df):
    #for test data
    df = df.fillna('NA')
    
    print("Shape of the data set before transforming : ", df.shape)
    
    #remove unncessary features
    df = df.drop(['zipcode','county','address','city','state'],axis=1)
    
    #transform longitude and latitude as discussed earlier
    df['x'] = df.latitude.apply(lambda x: math.cos(x)) * df.longitude.apply(lambda x : math.cos(x))
    df['y'] = df.latitude.apply(lambda x: math.cos(x)) * df.longitude.apply(lambda x : math.sin(x))
    df['z'] = df.latitude.apply(lambda x: math.sin(x))
    df = df.drop(['longitude','latitude'],axis=1)
    
    #get dummies for categorical features
    categorical_columns = ['time', 'property_type']
    for c in categorical_columns:
        temp = pd.get_dummies(df[c])
        df[temp.columns] = temp
    df = df.drop(categorical_columns, axis=1)
    
    print("Shape of the data set after transforming : ", df.shape,"\n")
    return df

def scaleMinMax(df,min,max):
    print("\nShape of the data set before min max scaling : ",df.shape)
    minMax = preprocessing.MinMaxScaler(feature_range=(min, max))
    np_scaled = minMax.fit_transform(df)
    df_scaled = pd.DataFrame(np_scaled,columns=df.columns)
    print("\nShape of the data set post min max scaling : ",df_scaled.shape)
    return df_scaled

def boxCoxTranformation(df):
    
    #assuming that only numerical features are presented
    
    print("Shape of the dataset before transformation : ", df.shape)
    
    print("Ignoring the columns....",list(df.select_dtypes(exclude=[np.number])))
    temp_df = df[list(df.select_dtypes(include=[np.number]))]
    print("Performing column transformations for :", list(temp_df) )
    
    #converting to positive for boxcox
    df_min_max = scaleMinMax(temp_df,1,2)
    #print(df_min_max)
    df_new = pd.DataFrame()
    for c in list(df_min_max):
        #print(c,df_min_max[c])
        temp_col_ser = df_min_max[c]
        #print(df_min_max[c].sort_values()[:5])
        #print(stats.boxcox(temp_col_ser)[0])
        df_new[c] = stats.boxcox(df_min_max[c])[0]
    
    print("Shape of the dataset after transformation : ", df_new.shape)
    return df_new

def newBoxCoxTranformation(df,target):
    
    #assuming that only numerical features are presented
    
    print("Shape of the dataset before transformation : ", df.shape)
    
    print("Ignoring the columns....",list(df.select_dtypes(exclude=[np.number])))
    temp_df = df[list(df.select_dtypes(include=[np.number]))]
    print("Performing column transformations for :", list(temp_df) )
    
    #converting to positive for boxcox
    scale_column = list(temp_df)
    scale_column.remove(target)
    df_min_max = scaleMinMax(temp_df[scale_column],1,2)
    #print(df_min_max)
    df_new = pd.DataFrame()
    for c in list(df_min_max):
        df_new[c] = stats.boxcox(df_min_max[c])[0]
    df_new['rent'] = df.rent.apply(lambda x: math.log(x))
    
    print("Shape of the dataset after transformation : ", df_new.shape)
    return df_new[scale_column], df_new['rent']
	
def returnTrainTestSet(df,frac=.7,random_state=200):
    
    print("Input data set shape : ",df.shape)
    all_columns = set(df.columns)
    all_columns.discard('rent')
    feature_columns = list(all_columns)
    lable_column = 'rent'
    
    print("\nfeature_columns : ", feature_columns)
    print("lable_column : ", lable_column)
    
    plot_column = feature_columns[random.randint(0,df.shape[1]-5)]
    df[plot_column].plot( kind='hist',title=plot_column)
    plt.show()
    df = scaleMinMax(df)
    
    
    df[plot_column].plot( kind='hist',title=plot_column)
    plt.show()
    
    df_train = df.sample(frac=frac,random_state=random_state).reset_index()
    df_test = df.drop(df_train.index).reset_index()
    print("\nTrain feature shape: ", df_train[feature_columns].shape, df_train[lable_column].shape)
    print("Test feature shape: ", df_test[feature_columns].shape, df_test[lable_column].shape)
    
    return df_train[feature_columns], df_train[lable_column], df_test[feature_columns], df_test[lable_column]
	
	

def printStats(test_result,test_y, printCounter = 0):
    
    #convert the inputs to array
    test_result = np.array(test_result)
    test_y = np.array(test_y)

    percentage_deviation = np.abs( ( test_result - test_y ) * 100 / test_y )
    
    cnt = Counter(percentage_deviation)
    
    less_1 = less_2 = less_3 = less_4 = less_5 = 0
    for c in cnt:
        if c < 1:
            less_1 += cnt[c]
        if c < 2:
            less_2 += cnt[c]
        if c < 3:
            less_3 += cnt[c]
        if c < 4:
            less_4 += cnt[c]
        if c < 5 :
            less_5 += cnt[c]

    print(np.round((less_1*100/test_y.size),2),"% of properties within predicted rent within 1% of actual rent")
    print(np.round((less_2*100/test_y.size),2),"% of properties within predicted rent within 2% of actual rent")
    print(np.round((less_3*100/test_y.size),2),"% of properties within predicted rent within 3% of actual rent")
    print(np.round((less_4*100/test_y.size),2),"% of properties within predicted rent within 4% of actual rent")
    print(np.round((less_5*100/test_y.size),2),"% of properties within predicted rent within 5% of actual rent")
    
    if printCounter:
        print(Counter(percentage_deviation))
    

def doubleCheckTheValidationScore(df, clf):
    
    X_train, y_train, X_test, y_test = returnTrainTestSet(df,.99995,250)
    
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1986)
    scores = cross_val_score(clf, X_train, y_train, cv=cv) 
    
    print("\nValidation Scores : ", scores)
    print("Mean : ",np.average(scores))
    print("Median : ",np.median(scores))
    print("Min : ",np.min(scores))
    print("Max : ",np.max(scores))



def plotPcaDf(df):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    ttt = df.drop('rent',axis=1)
    arr = pca.fit_transform(ttt)
    df_plot = pd.DataFrame(arr)
    df_plot.plot.scatter(0,1)


def getTrainingAndTestingScore():
    valid_score=[]
    test_score =[]
    for i in list(range(1,20,1)):
        my_model = XGBRegressor(max_depth=i)
        my_model.fit(X_train,y_train)
        
        valid_score.append(my_model.score(X_test,y_test))
        test_score.append(my_model.score(X,y))
        
        
        print("\nWith max_depth =",i,", Validation Score :",my_model.score(X_test,y_test), ", Testing Score :",my_model.score(X,y))
        print("As per Competetion Evaluation Metrics : ")
        printStats(my_model.predict(X),y)
        
    plt_df = pd.DataFrame({'max_depth':list(range(1,20,1)),
                            'valid_score' : valid_score,
                            'test_score' : test_score
                           })
    return plt_df

def plotDf(df,zeroBase=1):
    fig, ax = plt.subplots()
    df.plot.line('max_depth','valid_score',ax=ax, xticks=df.max_depth)
    df.plot.line('max_depth','test_score',ax=ax)

    vline = df[df.test_score == df.test_score.max()]['max_depth']
    if zeroBase == 0:
        ax.vlines(vline,df.test_score.min(),df.valid_score.max(),'r')
    else:
        ax.vlines(vline,0,df.valid_score.max(),'r')
    plt.show()

def getTrainingAndTestingScoreGrad():
    valid_score=[]
    test_score =[]
    for i in list(range(1,15,1)):
        my_model = GradientBoostingRegressor(max_depth=i)
        my_model.fit(X_train,y_train)
        
        valid_score.append(my_model.score(X_test,y_test))
        test_score.append(my_model.score(X,y))
        
        
        print("\nWith max_depth =",i,", Validation Score :",my_model.score(X_test,y_test), ", Testing Score :",my_model.score(X,y))
        print("As per Competetion Evaluation Metrics : ")
        printStats(my_model.predict(X),y)
        
    plt_df = pd.DataFrame({'max_depth':list(range(1,15,1)),
                            'valid_score' : valid_score,
                            'test_score' : test_score
                           })
    return plt_df

def getTrainingAndTestingScoreXGBoost():
    valid_score=[]
    test_score =[]
    for i in list(range(100,800,100)):
        my_model = XGBRegressor(max_depth=7,n_estimators=i)
        my_model.fit(X_train,y_train)
        
        valid_score.append(my_model.score(X_test,y_test))
        test_score.append(my_model.score(X,y))
        
        
        print("\nWith estimator =",i,", Validation Score :",my_model.score(X_test,y_test), ", Testing Score :",my_model.score(X,y))
        print("As per Competetion Evaluation Metrics : ")
        printStats(my_model.predict(X),y)
        
    plt_df = pd.DataFrame({'max_depth':list(range(100,800,100)),
                            'valid_score' : valid_score,
                            'test_score' : test_score,
                        })
    return plt_df

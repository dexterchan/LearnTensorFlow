from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
class SchemaPrep:
    CSV_COLUMNS = ['field0', 'field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8', 'field9', 'field10', 'field11', 'field12', 'field13', 'field14', 'field15', 'field16', 'field17', 'field18', 'field19', 'field20', 'field21', 'field22', 'field23', 'field24', 'field25', 'field26', 'field27', 'field28', 'field29', 'field30', 'field31', 'field32', 'field33', 'field34', 'field35', 'field36', 'field37', 'field38', 'field39', 'field40', 'field41', 'field42', 'field43', 'field44', 'field45', 'field46', 'field47', 'field48', 'field49', 'field50', 'field51', 'field52', 'field53', 'field54', 'field55', 'field56', 'field57', 'field58', 'field59', 'field60']
    CSV_COLUMN_DEFAULTS=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], ['']]
    LABELS=['M', 'R']
    LABEL_COLUMN = "field60"
    CATEGORICAL_COLS=()
    CONTINUOUS_COLS=['field0', 'field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8', 'field9', 'field10', 'field11', 'field12', 'field13', 'field14', 'field15', 'field16', 'field17', 'field18', 'field19', 'field20', 'field21', 'field22', 'field23', 'field24', 'field25', 'field26', 'field27', 'field28', 'field29', 'field30', 'field31', 'field32', 'field33', 'field34', 'field35', 'field36', 'field37', 'field38', 'field39', 'field40', 'field41', 'field42', 'field43', 'field44', 'field45', 'field46', 'field47', 'field48', 'field49', 'field50', 'field51', 'field52', 'field53', 'field54', 'field55', 'field56', 'field57', 'field58', 'field59']
    rows=0
    cols=0


    def __init__(self,csv_col, test_size, random_state):
        self.clear()

        #if(csv_col is not None):
        #    CSV_COLUMNS = csv_col
        if(test_size is not None):
            self.test_size=test_size

        if(random_state is not None):
            self.random_state=random_state

        return

    def clear(self):
        #self.CSV_COLUMNS=[]
        self.rows=0
        self.cols=0
        self.test_size=0.2
        self.random_state=200
        self.encoder=None
        return


    #assume the last column is label
    def read_DataSet(self,fileName):
        df = pd.read_csv(fileName)
        self.rows,self.cols = df.shape

        self.X = df[df.columns[0:self.cols-1]]#.values
        self.Y = df[df.columns[self.cols-1]]


        yy,self.LABELS=self.encodeLabel(self.Y)
        print("Labels:{0}".format(self.LABELS))
        X,Y = shuffle(self.X,self.Y,random_state=1)


        self.train_x,self.test_x,self.train_y,self.test_y = train_test_split(X,Y,test_size=self.test_size, random_state=self.random_state)


        self.defineDefaultLabel(fileName)


        return

    def defineDefaultLabel(self,fileName):
        df = pd.read_csv(fileName, nrows=2)
        rows,cols = df.shape
        fRange = np.arange(cols)


        self.CSV_COLUMNS=["field"+str(i) for i in fRange]
        #or using map
        #map(lambda x: 'field'+str(x), r)
        sampleRow = list(df.loc[0,:])
        import numbers
        self.CSV_COLUMN_DEFAULTS = map ( lambda x: [0.0] if isinstance(x,numbers.Number) else [""] ,sampleRow)

        #assume last column is data
        self.LABEL_COLUMN="field"+str(cols-1)

        self.CONTINUOUS_COLS = ["field"+str(i) for i in np.arange(cols-1) ]

        print (self.CSV_COLUMNS)
        print(self.CSV_COLUMN_DEFAULTS)
        print(self.LABEL_COLUMN)
        print(self.LABELS)
        print(self.CONTINUOUS_COLS)
        return

    def encodeLabel(self, Y):
        self.encoder = LabelEncoder()
        self.encoder.fit(Y)
        yy = self.encoder.transform(Y)
        LABELS = self.encoder.classes_.tolist()
        return yy,LABELS

    def outputDataSet(self,filePrefix):

        outputBuffer=None


        outputBuffer=(pd.concat([self.train_x,self.train_y],axis=1))
        trainFileName=str(filePrefix) + str(".train.csv")
        outputBuffer.to_csv(trainFileName,index=False)

        outputBuffer=(pd.concat([self.test_x,self.test_y],axis=1))
        testFileName=str(filePrefix) + str(".test.csv")
        outputBuffer.to_csv(testFileName,index=False)

        return

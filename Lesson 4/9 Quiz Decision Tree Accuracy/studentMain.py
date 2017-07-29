import sys
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

acc =clf.score(features_test, labels_test)
    
def submitAccuracies():
  return {"acc":round(acc,3)}

if __name__ == '__main__':
    print submitAccuracies()
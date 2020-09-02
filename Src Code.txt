import pandas as pan
import numpy as nump
import scipy.stats as scistats

def labellize(diagnosis):
    if diagnosis == 'B':
        return 0
    else:
        return 1



cancerdata = pan.read_csv('data.csv')
cancerdata=cancerdata.drop(['id', 'Unnamed: 32'], axis=1)
cancerdata['diagnosis'] = cancerdata['diagnosis'].apply(labellize)
cancerdata = cancerdata.rename(columns={"diagnosis": "class"})
cancerdata=cancerdata[['radius_worst','concave points_mean','area_worst','area_mean','concave points_worst','perimeter_mean',
                 'area_se','concavity_worst','radius_se','compactness_worst','class']]





def InfoGain(data, split_attribute_name, class_name="class"):
    # Calculate the calc_entr of the total cancerdata
    total_calc_entr = calc_entr(data[class_name])

    #Calculating calc_entr of cancerdata

    #Calculating values and  corresponding counts of the split attribute
    vals, counts = nump.unique(data[split_attribute_name], return_counts=True)

    #Calculating weighted calc_entr
    Weighted_calc_entr = nump.sum(
        [(counts[i] / nump.sum(counts)) * calc_entr(data.where(data[split_attribute_name] == vals[i]).dropna()[class_name])
         for i in range(len(vals))])

    #Calculating information gain
    InfGain = total_calc_entr - Weighted_calc_entr
    return InfGain


def calc_entr(class_col):
    elements, counts = nump.unique(class_col, return_counts=True)
    calc_entr = nump.sum(
        [(-counts[i] / nump.sum(counts)) * nump.log2(counts[i] / nump.sum(counts)) for i in range(len(elements))])
    return calc_entr


def ID3(data, originaldata, features, classname="class", classofparent=None):
    # Now Defining stopping criteria => when one of these is satisfied, we return a leaf node

    # When all the class_values have the same value, return the value
    if len(nump.unique(data[classname])) <= 1:
        return nump.unique(data[classname])[0]

    #When the cancerdata is empty, we return mode class feature value of the original cancerdata
    elif len(data) == 0:
        return nump.unique(originaldata[classname])[
            nump.argmax(nump.unique(originaldata[classname], return_counts=True)[1])]

    #If the feature space is empty, return the mode class feature value of the direct parent node

    elif len(features) == 0:
        return classofparent

    #If none of these holds true, then we grow the tree

    else:
        # Set the default value for this node --> The mode class feature value of the current node
        classofparent = nump.unique(data[classname])[
            nump.argmax(nump.unique(data[classname], return_counts=True)[1])]

        #Implementing the subspace sampling.

        features = nump.random.choice(features, size=nump.int(nump.sqrt(len(features))), replace=False)

        # Select the feature which best splits the cancerdata
        item_values = [InfoGain(data, feature, classname) for feature in
                       features]  # Return the information gain values for the features in the cancerdata
        optimumfeature_index = nump.argmax(item_values)
        optimumfeature = features[optimumfeature_index]

        # Create the tree structure. The root gets the name of the feature (optimumfeature) with the maximum information
        # gain in the first run
        tree = {optimumfeature: {}}

        # Remove the feature with the best information gain from the feature space
        features = [i for i in features if i != optimumfeature]

        # Grow a branch under the root node for each possible value of the root node feature

        for value in nump.unique(data[optimumfeature]):
            value = value
            # Split the cancerdata along the value of the feature with the largest information gain and then create sub_cancerdatas
            subsampledata = data.where(data[optimumfeature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_cancerdatas with the new parameters --> Here the recursion comes in!
            subtree = ID3(subsampledata, cancerdata, features, classname, classofparent)

            # Add the sub tree, grown from the sub_cancerdata to the tree under the root node
            tree[optimumfeature][value] = subtree

        return (tree)

def pred_fromdata(search_qn, tree, default=0):
    for key in list(search_qn.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][search_qn[key]]
            except:
                return default
            result = tree[key][search_qn[key]]
            if isinstance(result, dict):
                return pred_fromdata(search_qn, result)
            else:
                return result
#
def classname(cancerdata):
    train_set = cancerdata.iloc[:round(0.8 * len(cancerdata))].reset_index(
        drop=True)  # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    test_set = cancerdata.iloc[round(0.8 * len(cancerdata)):].reset_index(drop=True)
    return train_set, test_set


train_set = classname(cancerdata)[0]
test_set = classname(cancerdata)[1]

#*Training the RF model (random_forest)
def RF_Train(cancerdata, number_of_Trees):
    # Create a list in which the single forests are stored
    RFsubtree = []

    # Create a number of n models
    for i in range(number_of_Trees):
        # bootstrapping sampled cancerdatas from the original cancerdata
        bootstrap_sample = cancerdata.sample(frac=1, replace=True)

        #calling the classname function to split into training and testing cancerdatas
        boottrainset = classname(bootstrap_sample)[0]
        boottestset = classname(bootstrap_sample)[1]

        #Growing a tree model from train data and appending a tree created from bootstrapping to whole random forest
        RFsubtree.append(ID3(boottrainset, boottrainset,
                                          boottrainset.drop(labels=['class'], axis=1).columns))

    return RFsubtree

#Predicting for given search_qn#
def RF_Predict(search_qn, random_forest, default=0):
    predictions = []
    for tree in random_forest:
        predictions.append(int(pred_fromdata(search_qn, tree, default)))
    #print(predictions)
    return scistats.mode(predictions)[0][0]


search_qn = test_set.iloc[0, :].drop('class').to_dict()
search_qn_class = test_set.iloc[0, 0]
#Individual predictions can be checked for each of the test data provided from testing data splitted
#prediction = RF_Predict(search_qn, random_forest)

#Test RF on test data sample and print the test_accuracy#
def RF_Test(data, random_forest):
    data['predictions'] = None
    for i in range(len(data)):
        search_qn = data.iloc[i, :].drop('class').to_dict()
        data.loc[i, 'predictions'] = RF_Predict(search_qn, random_forest, default=0)
    test_accuracy = sum(data['predictions'] == data['class']) / len(data) * 100
    print('The prediction accuracy using the implemented Random Forests (without SKLearn) is: ',sum(data['predictions'] == data['class'])/len(data)*100,'%')
    return test_accuracy

#*Call the function RF_Test
#no_trees=[5,10,20,25,50]
#for i in no_trees:
#The number of trees can be changed here
#    print("\n\n\nnumber of trees in the forest : " + str(i))
#    random_forest = RF_Train(cancerdata, i)
#The built random forest is used for testing (25% of the data is split into testing while 75% is training
#    RF_Test(test_set, random_forest)

random_forest = RF_Train(cancerdata, 10)
RF_Test(test_set, random_forest)
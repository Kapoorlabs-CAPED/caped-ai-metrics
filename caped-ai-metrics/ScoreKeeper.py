from scipy import spatial
import pandas as pd
import numpy as np
from skimage import measure 

class ClassificationScore:
    
    def __init__(self, predictions, groundtruth, segimage, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):

         #A list of all the prediction csv files, path object
         self.predictions = predictions 
         #Segmentation image for accurate metric evaluation
         self.segimage = segimage
         #Approximate locations of the ground truth, Z co ordinate wil be ignored
         self.groundtruth = groundtruth
         self.thresholdscore = thresholdscore
         self.thresholdspace = thresholdspace 
         self.thresholdtime = thresholdtime
         self.location_pred = []
         self.location_gt = []

         self.dicttree = {}

    def make_trees(self):
        
        for i in range(self.segimage.shape[0]):
            #Make a ZYX image
            currentimage = self.segimage[i,:,:,:]
            props = measure.regionprops(currentimage)
            indices = [prop.centroid for prop in props]
            if len(indices) > 0:
                tree = spatial.cKDTree(indices)
                self.dicttree[int(i)] = [tree, indices] 
             

    def model_scorer(self):

         Name = []
         TP = []
         FP = []
         FN = []
         GT = []
         Pred = []
         columns = ['Model Name', 'True Positive', 'False Positive', 'False Negative', 'Total Predictions', 'GT predictions']
         

         dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
         for index, row in dataset_gt.iterrows():
              T_gt = int(row[0])
              current_point = (row[1], row[2], row[3])
              tree, indices = self.dicttree[int(T_gt)]
              distance, nearest_location = tree.query(current_point)
              nearest_location = (int(indices[nearest_location][0]), int(indices[nearest_location][1]), int(indices[nearest_location][2]))
              self.location_gt.append([T_gt, nearest_location[0], nearest_location[1], nearest_location[2]])
        
         

         for csv_pred in self.predictions:
            self.location_pred = []
            self.listtime_pred = []
            self.listy_pred = []
            self.listx_pred = []
            self.listscore_pred = []
            self.csv_pred = csv_pred
            name = self.csv_pred.stem
            dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')

            for index, row in dataset_pred.iterrows():
              T_pred = int(row[0])
              current_point = (row[1], row[2], row[3])
              score = row[4]
              if score >= float(self.thresholdscore): 
                  self.location_pred.append([int(T_pred), int(row[1]), int(row[2]), int(row[3])])
              
            tp, fn, fp, pred, gt = self.TruePositives()
            
            Name.append(name)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            GT.append(gt)
            Pred.append(pred)
         data = list(zip(Name, TP, FP, FN, Pred, GT))

         df = pd.DataFrame(data, columns=columns)
         df.to_csv(str(self.csv_pred.parent) + '_Model_Accuracy' + '.csv')
         return df

     

    def TruePositives(self):

            tp = 0
            fp = 0
            tree = spatial.cKDTree(self.location_gt)
            for i in range(len(self.location_pred)):
                
                return_index = self.location_pred[i]
                closestpoint = tree.query(return_index)
                spacedistance, timedistance = TimedDistance(return_index, self.location_gt[closestpoint[1]])
                    
                if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                        tp  = tp + 1
                else:
                        fp = fp + 1        
            
            fn = self.FalseNegatives()
            return tp, fn, fp, len(self.location_pred), len(self.location_gt)
        

    def FalseNegatives(self):
        
                        tree = spatial.cKDTree(self.location_pred)
                        fn = 0
                        for i in range(len(self.location_gt)):
                            
                            return_index = (int(self.location_gt[i][0]),int(self.location_gt[i][1]),
                                            int(self.location_gt[i][2]), int(self.location_gt[i][3]))
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = TimedDistance(return_index, self.location_pred[closestpoint[1]])

                            if spacedistance > self.thresholdspace or timedistance > self.thresholdtime:
                                    fn  = fn + 1

                        return fn
                    
                                
def EuclidMetric(x,y):
    
    return (x - y) * (x - y) 

def MannhatanMetric(x,y):
    
    return np.abs(x - y)

def EuclidSum(func):
    
    return float(np.sqrt( np.sum(func)))

def ManhattanSum(func):
    
    return float(np.sum(func))

def general_dist_func(metric):
     
     return lambda x,y : [metric(x[i], y[i]) for i in range(1,len(x))]
 
def TimedDistance(pointA, pointB):

     dist_func = general_dist_func(EuclidMetric)
     
     spacedistance = EuclidSum(dist_func(pointA, pointB))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     return spacedistance, timedistance
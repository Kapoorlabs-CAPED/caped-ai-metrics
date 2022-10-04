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

         self.listtime_pred = []
         self.listy_pred = []
         self.listx_pred = []
         self.listscore_pred = []

         self.listtime_gt = []
         self.listz_gt = []
         self.listy_gt = []
         self.listx_gt = []
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
              T_gt = row[0]
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
              T_pred = row[0]
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
                        for i in range(len(self.listtime_gt)):
                            
                            return_index = (int(self.listtime_gt[i]),int(self.listy_gt[i]), int(self.listx_gt[i]))
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = TimedDistance(return_index, self.location_pred[closestpoint[1]])

                            if spacedistance > self.thresholdspace or timedistance > self.thresholdtime:
                                    fn  = fn + 1

                        return fn
                    
                    
    
                    
                    
                                
 
def TimedDistance(pointA, pointB):

    
     spacedistance = float(np.sqrt( (pointA[1] - pointB[1] ) * (pointA[1] - pointB[1] ) + (pointA[2] - pointB[2] ) * (pointA[2] - pointB[2] )  ))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     
     return spacedistance, timedistance
from matplotlib import pyplot as plt
import numpy as np, cv2, math, glob
import csv


def convolution(ip_img, im_rows, im_cols, sobel_mask):
    #convultion operation on image using sobel operator
    if im_rows>=3 and im_cols>=3:
        res_img = np.zeros((im_rows, im_cols))
        for imr in range(1, im_rows-1):
            for imc in range(1, im_cols-1):
                res_img[imr][imc] = np.sum(sobel_mask*ip_img[imr-1:imr+2, imc-1:imc+2]) #convolution of image using sobel operator
                res_img[imr][imc] /= 4 # normalizing each pixel for the sobel operator
        return res_img
    else:
        print("Sobel Operation could not be performed as size of the image is not greater than or equal to size of Sobel Operator!")
        return ip_img
    
def HOGFeatures(cell_res, rows, cols):
    #Computing HOG features for the image
    #This returns 
    hog_feature_list = list()
    for i in range(rows-1):
        for j in range(cols-1):
            blocks = list()
            blocks = blocks + list(cell_res[i][j]) + list(cell_res[i][j+1]) + list(cell_res[i+1][j]) + list(cell_res[i+1][j+1])
            cal_val = math.sqrt(sum(i ** 2 for i in blocks))
            
            if cal_val != 0:
                blocks = [x/cal_val for x in blocks]
            hog_feature_list.append(blocks) 
            #Appending all the values to a histogram list
    return [colss for rowss in hog_feature_list for colss in rowss]

    
def cellHistogram(mag_mat, ang_mat):
    #Classifying the gradiant angle using bin centers and assigning gradiant magnitude values to the histogram list
    nr_a, nc_a, histo = ang_mat.shape[0],ang_mat.shape[1], np.zeros(9, dtype=np.float32)
    for r in range(0, nr_a):
        for c in range(0, nc_a):           
            val = int(ang_mat[r, c])
            if val < 0:
                idx1, idx2 = 8, 0
                cal = (idx2*20 - ang_mat[r, c])/20
                histo[idx1] = histo[idx1] + (mag_mat[r, c]*(cal))
                histo[idx2] = histo[idx2] + (mag_mat[r, c]*(1-cal))
                
            elif val >= 160:
                idx1, idx2 = 0, 8
                cal = (ang_mat[r, c] - idx2*20)/20
                histo[idx1] = histo[idx1] + (mag_mat[r, c]*(cal))
                histo[idx2] = histo[idx2] + (mag_mat[r, c]*(1-cal))
                
            else:
                idx1, idx2 = int(ang_mat[r, c]// 20), int(ang_mat[r, c]// 20 + 1)
                cal = (idx2*20 - ang_mat[r, c])/20
                histo[idx1] = histo[idx1] + (mag_mat[r, c]*(cal))
                histo[idx2] = histo[idx2] + (mag_mat[r, c]*(1-cal))
    return histo

def imageProcess(imgy):
    
    # Sobel Operator - Gx
    sobel_gx = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]]) 
    # Sobel Operator - Gy
    sobel_gy = np.array([[1, 2, 1], 
                         [0, 0, 0], 
                         [-1, -2, -1]]) 
    
    # computing horizontal gradient and vertical gradient using sobel operator
    op_gx, op_gy  = convolution(imgy, imgy.shape[0], imgy.shape[1], sobel_gx), convolution(imgy, imgy.shape[0], imgy.shape[1], sobel_gy)
    # computing gradient angle by using inverse tan and computing magnitude using horizontal and vertical gradient
    tan_val = np.arctan2(op_gy, op_gx)
    op_magnitude = (np.sqrt((abs(op_gy) * abs(op_gy)) + (abs(op_gx) * abs(op_gx)))/math.sqrt(2)) #normalise the magnitude by root(2)   
    op_angle = tan_val* (180/np.pi)
    
#    cv2.imwrite('Gradient_Magnitude.bmp', op_magnitude)  gradiant for test for test images
    
    temp_angle = np.copy(op_angle)
    nrows, ncols = temp_angle.shape[0], temp_angle.shape[1]
    for n_r in range(0, nrows):
        for n_c in range(0, ncols):
            if temp_angle[n_r,n_c]<-10: temp_angle[n_r,n_c] += 360 # making angle positive
            if temp_angle[n_r,n_c] >= 170 and temp_angle[n_r,n_c] < 350: temp_angle[n_r,n_c] -= 180


    # creating cells 
    size_cell, cell_res = 8, []
    
    for i in range(0, nrows , size_cell):
        op_column_wise = []
        for j in range(0, ncols , size_cell):
            hist_cell = cellHistogram(op_magnitude[i:i + size_cell, j:j + size_cell], temp_angle[i:i + size_cell, j:j + size_cell])
            op_column_wise.append(np.array(hist_cell,dtype=np.float32))
        cell_res.append(op_column_wise)
    si = np.shape(cell_res)
    hog_features = HOGFeatures(cell_res, si[0], si[1])
    return hog_features

def LocalBinaryPartition(imgy):
    #Performing local binary partition measure
    r_im, c_im = imgy.shape[0], imgy.shape[1]
    bin_pat, var = np.zeros((r_im,c_im)), np.zeros((16,16))
    final_res = list()
    list_uniform = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
    
    for rows in range(0, r_im - 15, 16):
        for cols in range(0, c_im - 15, 16):
            block_hist, var = [0] * 59, imgy[rows:rows+16][cols:cols+16]
            for strt in range(rows, rows+16):
                for end in range(cols, cols+16):
                    bin_str = ''
                    #Appending 0/1 character by character to a string
                    if (end == 0) or (end == c_im-1) or (strt == 0) or (strt == r_im-1):
                        bin_pat[strt][end] = 5
                    else:
                        if imgy[strt-1][end-1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
            
                        if imgy[strt-1][end] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt-1][end+1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt][end+1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt+1][end+1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt+1][end] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt+1][end-1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        
                        if imgy[strt][end-1] <= imgy[strt][end]: bin_str = bin_str + '0'
                        else: bin_str = bin_str + '1'
                        #Converting the string value to binary
                        bin_pat[rows][cols] = int(bin_str, 2)
                        
                        #Assigning the integer values to one of 59 bins
                        if bin_pat[rows][cols] not in list_uniform: block_hist[58] += 1
                        else: block_hist[list_uniform.index(bin_pat[rows][cols])] += 1
            new_block_hist = [float(val) for val in block_hist]
            final_res.extend(new_block_hist)
            
    return final_res  

def dataSetCreation(path, flag):
    #This method is used to call the HOG nad LBP features calculation methods for each image 
    print("reading files from " + str(path))
    if flag == 0:
        data_list = []
        img_list = glob.glob(path)
        for img_name in img_list:
            imgy = cv2.imread(img_name)
            imgy = np.round(0.299 * imgy[:,:,0] + 0.587 * imgy[:,:,1] + 0.114 * imgy[:,:,2])
            hog_features = imageProcess(imgy)
            data_list.append(hog_features)
        return data_list
    if flag == 1:
        data_list = []
        img_list = glob.glob(path)
        for img_name in img_list:
            imgy = cv2.imread(img_name)
            imgy = np.round(0.299 * imgy[:,:,0] + 0.587 * imgy[:,:,1] + 0.114 * imgy[:,:,2])
            hog_features = imageProcess(imgy)
            lbp_features = LocalBinaryPartition(imgy)       
            data_list.append(hog_features+lbp_features)
        return data_list
    return None

    
        
    
class Perceptron_2layer_Model(object):
    #MAKE THE VARIABLE CHANGES ACCORDING TO THE OUTPUT EXECUTED
    def __init__(self):   
        self.inp_layer_size, self.hidden_layer_size, self.output_layer_size =  11064, 400, 1   #7524, 200, 1   
        np.random.seed(1)
        self.wt1 = np.random.uniform(low=0, high=1, size=(11064,400))      #(7524,400)  OR (7524,200) OR (11064,200)
        self.wt2 = np.random.uniform(low=0, high=1, size=(400,1))          # (200,1)

        self.bias1 = np.random.uniform(low=0, high=1, size=(400,1))        #(200,1)
        self.bias2 = np.random.uniform(low=0, high=1, size=(1,1))

    #Method for forward propagation
    def propForward(self, data):        
        self.updated_val = np.dot(data, self.wt1) + self.bias1.transpose() 
        self.updated_val[self.updated_val<=0] = 0
        self.val2 = np.dot(self.updated_val, self.wt2)+self.bias2 
        self.val3 = 1/(1+np.exp(-self.val2))     
        return self.val3 

    #Method for backward propagation and training the model
    def propBack(self, inp, labels, output):
        
        cal_error = self.output - labels
        self.op_error = cal_error
        
        sig_err = 1/(1+np.exp(-self.output))
        typ1_err =  np.multiply(self.op_error, sig_err*(1-sig_err))
        
        self.updated_val[self.updated_val>0] = 1
        self.updated_val[self.updated_val<=0] = 0 
        
        typ2_error = np.multiply(typ1_err, self.wt2.transpose())*self.updated_val   
        new_wt1, new_wt2 = np.dot(inp.transpose(), typ2_error), np.dot(self.updated_val.transpose(), typ1_err)
        
        self.wt1, self.wt2 = self.wt1+(-0.01*new_wt1), self.wt2+(-0.01*new_wt2)
        self.bias1, self.bias2 = self.bias1 + (-0.01*typ2_error.transpose()), self.bias2 + (-0.01*typ1_err)
        
    #Training the perceptron network
    def training(self, train_data, label_data):
        self.output = self.propForward(train_data)
        self.propBack(train_data, label_data, self.output)
    
    # predicting the test images
    def prediction(self, data_predict):
        print("prediction for:\n", str(data_predict))
        ans = self.propForward(data_predict)
        print(str(ans)," predicted value")     
        
    # testing the model for accuracy
    def testing(self, x_test, y_test):
        res, ct = [], 0
        ans_final =  self.propForward(x_test)
        print("Prediction for all test images: \n", ans_final)
        for sample in range(0, len(ans_final)):
            if ans_final[sample] >= 0.6:
                op = 1
                res.append(op)
            elif ans_final[sample] <= 0.4:
                op = 0
                res.append(op)
            else:
                op = 0.5

            if op==y_test[sample]:
                ct = ct + 1
                
        print("Accuracy is: ",str((float(ct)/float(len(y_test)))*100))
        
    def start_training(self, train_x, train_y, epochs, input_size):
        for i in range(epochs+1):# Perceptron is trained 500/1000 times
            for j in range(20):
                mod_x = np.reshape(train_x[j],(input_size,1))
                mod_y = np.reshape(train_y[j],(1,1))
                self.training(mod_x.transpose(), mod_y)
            if i%10==0:
                print("Epochs: {}, Loss: {}".format(i,str(np.mean(np.square(train_y - self.propForward(train_x))))))

def main():

    # UNCOMMENT THE BELOW BLOCK OF CODE TO RUN FOR ONLY HOG FEATURES
    '''
    # HOG FEATURES 
    train_pos = dataSetCreation(r"Trpos/*.bmp", 0)
    test_pos = dataSetCreation(r"tepos/*.bmp", 0)
    train_neg = dataSetCreation(r"Trneg/*.bmp", 0)
    test_neg = dataSetCreation(r"teneg/*.bmp", 0)
    
    train_x = np.asarray(np.concatenate([train_pos, train_neg]))
    train_y = np.array(([1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]))
    
    test_x = np.asarray(np.concatenate([test_pos,test_neg]))
    test_y = np.array(([1], [1], [1], [1], [1], [0], [0], [0], [0], [0]))
    
    
    # Training the Model
    
    ml_hog = Perceptron_2layer_Model()
    epochs = 1000
    input_size = 7524
    ml_hog.start_training(train_x, train_y, epochs, input_size) #This is the training function which is commented as the network is already trained.
    ml_hog.testing(test_x, test_y)
    '''
    # HOG + LBP FEATURES 
    #Add path to teh input files accordingly
    train_pos = dataSetCreation(r"Trpos/*.bmp", 1)
    test_pos = dataSetCreation(r"tepos/*.bmp", 1)
    train_neg = dataSetCreation(r"Trneg/*.bmp", 1)
    test_neg = dataSetCreation(r"teneg/*.bmp", 1)
    #print(len(train_pos))
    #print(len(test_pos))
    #print(train_pos[0])
    #print(len(train_pos[0]))
    
    train_x = np.asarray(np.concatenate([train_pos, train_neg]))
    train_y = np.array(([1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]))
    
    test_x = np.asarray(np.concatenate([test_pos,test_neg]))
    test_y = np.array(([1], [1], [1], [1], [1], [0], [0], [0], [0], [0]))
    
    #print("X "+train_x.shape)        
    #print("Y "+train_y.shape)
    # Training the Model  
    ml_hog_lbp = Perceptron_2layer_Model()
    epochs = 500
    input_size = 11064
    
    # For training ####### uncomment these lines
    ml_hog_lbp.start_training(train_x, train_y, epochs, input_size) #This is the training function which is commented as the network is already trained.
    ml_hog_lbp.testing(test_x, test_y)
                

main()

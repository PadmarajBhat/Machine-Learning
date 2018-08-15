import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
#Mat Dataset & The Paper
#http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381

#what is our data? let us see 3 tumor images and see the tumor section

#tumor list
meningioma = []
glioma = []
pituitary = []

#test list for the final test
test_features = []
test_labels = []
test_fid = []

#validation list for each epoch
valid_features = []
valid_labels = []


#take a backup copy
meningioma_bk = []
glioma_bk = []
pituitary_bk = []

last_batch = []

def reload_input_data():
    global meningioma_bk, glioma_bk, pituitary_bk
    
    meningioma_bk = meningioma[:]
    glioma_bk = glioma[:]
    pituitary_bk = pituitary[:]
    
    return

def check_if_batch_is_feasible(batch_size):
    return min( len(meningioma_bk), len(glioma_bk), len(pituitary_bk) ) > int(batch_size/3)

def open_mat_file(folder,fid):
    file_name = r'%s/%d.mat' %(folder,fid)
    #file_name = r'C:\Users\padmaraj.bhat\Desktop\Deep Learning\Final Project\brainTumorDataPublic_1-766\%d.mat' %(file_id,)
    f = h5py.File(file_name,'r')
    return f

def get_tumor_neighbor(a,o,k1):
    
    k = k1[:]
    #print("shape of k : ", len(k))
    b = np.zeros((len(k),a.shape[0], a.shape[1]))
    b[0] = o
    del k[-1]
    for i in range(a.shape[0] ):
        if np.count_nonzero(a[i]) > 0:
            for j in range(a.shape[1] ):
                if a[i][j] != 0 :#and b[i][j] == 0:
                    l = 1
                    for m in k:
                        if i-m >= 0 and j+m <= 512 :
                            b[l, i-m:i, j:j+m] = o[i-m:i, j:j+m]
                        if i-m >= 0 and j-m >= 0:
                            b[l, i-m:i,j-m:j] = o[i-m:i,j-m:j]
                        if i+m <=512 and j-m >= 0:
                            b[l, i:i+m, j-m:j] = o[i:i+m, j-m:j]
                        if i+m <=512 and j+m <=512:
                            b[l, i:i+m, j:j+m] = o[i:i+m, j:j+m]
                        l+=1
    
    return b

def rotate_the_image(orig_image_1):
    #plt.gcf().clear()
    #plt.close('all')
    
    
    #plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.subplot(1,4,1,title="0")
    plt.imshow(np.rot90(orig_image_1,0))
    plt.subplot(1,4,2,title="90")
    plt.imshow(np.rot90(orig_image_1,1))
    #plt.show()
    plt.subplot(1,4,3,title="180")
    plt.imshow(np.rot90(orig_image_1,2))
    #plt.show()
    plt.subplot(1,4,4,title="270")
    plt.imshow(np.rot90(orig_image_1,3))
    plt.tight_layout()
    plt.show()
    plt.close('all')
    #plt.gcf().clear()
    
def display_random_data():

    #file_id = random.randint(1533,2298)
    #f = open_mat_file("brainTumorDataPublic_1533-2298",file_id)
    file_id = random.randint(1,3064)
    
    folder = ""
    range_start = 0
    range_end = 0
    if file_id <= 766:
        range_start = 1
        range_end = 766
        folder = "brainTumorDataPublic_1-766"
    elif file_id <= 1532:
        range_start = 767
        range_end = 1532
        folder = "brainTumorDataPublic_767-1532"
    elif file_id <= 2298:
        range_start = 1533
        range_end = 2298
        folder = "brainTumorDataPublic_1533-2298"
    else:
        range_start = 2299
        range_end = 3064
        folder = "brainTumorDataPublic_2299-3064"
    
    f = open_mat_file(folder,file_id)
    
    #fetch data from the file
    tumor_mri = np.array(f['cjdata']['tumorMask'])
    orig_image = np.array(f['cjdata']['image'])
    orig_image = orig_image/orig_image.max()
    label = np.array(f['cjdata']['label'],np.int)
    
    pid_2_str=""
    pid_2 = np.array(f['cjdata']['PID'])
    for i in range (pid_2.shape[0]):
            pid_2_str += str(pid_2[i][0])

    
    #tumor_neighbors = get_tumor_neighbor(tumor_mri, orig_image, 20 )
    tumor_names = ['meningioma', 'glioma', 'pitutory']

    plt.subplot(1,2,1,title=tumor_names[label[0][0] -1] + " Image with tumor region highlighted")
    plt.imshow(orig_image)
    plt.subplot(1,2,2)

    plt.imshow(orig_image)
    plt.imshow(tumor_mri, alpha=0.3)
    plt.show()
    plt.close('all')

    print("This is how the data is fed to CNN: ")

    
    #rotate_the_image(orig_image)
    transformed_images = get_tumor_neighbor(tumor_mri, orig_image, [100, 75, 50, 40, 32, 24, 16, 8, 4, 1, 0] )
    caption = ["original image with the tumor", "100 neighbors", "75 neighbors", "50 neighbors", "40 neighbors", "32 neighbors", \
              "24 neighbors","16 neighbors", "8 neighbors", "4 neighbors", "1 neighbor",\
              ]
    for image_1 in range (transformed_images.shape[0]):
        if image_1 == 0:
            print(caption[image_1] + " and its 3 rotations")
            rotate_the_image(transformed_images[image_1])
        else:
            print("Tumor with "+ caption[image_1] + " and its 3 rotations")
            rotate_the_image(transformed_images[image_1])
        
    
    j=1
   
    print("\nFor the Same Patient Diff MRI Images Available are as Below: \n")
    
    for i in range(range_start,range_end):
        f = open_mat_file(folder,i)

        pid_1 = np.array(f['cjdata']['PID'])
        pid=""


        for i in range (pid_1.shape[0]):
            pid += str(pid_1[i][0])
        
        if float(pid) == float(pid_2_str):
            image_data = np.array(f['cjdata']['image'])
            
            #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.subplot(1,3,j)
            plt.imshow(image_data)
            
            j += 1
            if j > 3:
                j = 1
                plt.tight_layout()
                plt.show()
                plt.gcf().clear()
                plt.close('all')

    plt.tight_layout()
    plt.show()
    
    plt.gcf().clear()
    plt.close('all')

    #return orig_image, tumor_mri
    
def create_list(folder,start_i,end_i):
   
    #read the file and create seperate list for tumors
    for i in (range(start_i,end_i,1)):
        f = open_mat_file(folder,i) 
        y = np.array(f['cjdata']['label'],np.int)
        image_data = np.array(f['cjdata']['image'],np.float64)
        
        if image_data.shape[0] == 512:
            if y[0][0] == 1:
                meningioma.append( [folder, i, 0] )

            elif y[0][0] == 2:
                glioma.append( [folder, i, 1] )

            elif y[0][0] == 3:
                pituitary.append( [folder, i, 2] )
        else:
            print("Skipping: ", folder, " FileID: ", i, "of image shape : ",image_data.shape, "Tumor : ", y[0][0] )
            
    #shuffle the data before fetching test features
    random.shuffle(meningioma)
    random.shuffle(glioma)
    random.shuffle(pituitary)
    
    #let us take out the testing set out 
    print ("meningioma : ", np.array(meningioma).shape,\
           "glioma : ", np.array(glioma).shape, \
           "pituitary : ", np.array(pituitary).shape,"\n")
    
    #return [meningioma, glioma, pituitary]

    


def create_test_set(test_size):

    min_data_set = min(len(pituitary), len(glioma), len(meningioma))
    test_data_size = int(min_data_set * test_size / 100)
    
    for i in range(test_data_size):
        random.shuffle(meningioma)
        random.shuffle(glioma)
        random.shuffle(pituitary)
        test_fid.append(meningioma[i])
        test_fid.append(glioma[i])
        test_fid.append(pituitary[i])
        del meningioma[i]
        del glioma[i]
        del pituitary[i]
    
    

def get_image(folder,fid,size):

    f = open_mat_file(folder,fid) 
    
    tumor_mri = np.array(f['cjdata']['tumorMask'])
    orig_image = np.array(f['cjdata']['image'],dtype=np.float128)
    
    orig_image = orig_image/orig_image.max()
    
    if type(size) == list:
        tumor_neighbors = get_tumor_neighbor(tumor_mri, orig_image, size )
        return tumor_neighbors
    else:
        return orig_image

    #return tumor_neighbors[:, :, np.newaxis]
    
    



def one_hot_encode(x):
    b=[]

    import sklearn.preprocessing
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(int(max(x))))
    b = label_binarizer.transform(x)
   
    return b

def get_a_batch(a_batch_size, kneighb):

    batch_features = []
    batch_labels = []
    
    global last_batch
    last_batch = []
    
    transormed_a_batch_size = int(float(a_batch_size)/3)
    
    if min( len(meningioma_bk), len(glioma_bk), len(pituitary_bk) ) < transormed_a_batch_size:
        return [np.array(batch_features), np.array(batch_labels_transformed)]
    
    batch_index = 0
    for i in range(transormed_a_batch_size):
        
        '''print ("meningioma : ", np.array(meningioma_bk).shape,\
       "glioma : ", np.array(glioma_bk).shape, \
       "pituitary : ", np.array(pituitary_bk).shape,"\n")'''
        
        #print (np.array(batch_features).shape, np.array(batch_labels).shape)
        random.shuffle(meningioma_bk)
        random.shuffle(glioma_bk)
        random.shuffle(pituitary_bk)
        
        if type (kneighb)  == int:
            batch_features.append( get_image(meningioma_bk[0][0],meningioma_bk[0][1],0) )
            batch_labels.append( (meningioma_bk[0][2]) )

            batch_features.append( get_image(glioma_bk[0][0],glioma_bk[0][1],0) )
            batch_labels.append( (glioma_bk[0][2]) )


            batch_features.append( get_image(pituitary_bk[0][0],pituitary_bk[0][1],0) )
            batch_labels.append( pituitary_bk[0][2])  
        else:
            batch_features.append( get_image(meningioma_bk[0][0],meningioma_bk[0][1],kneighb) )
            batch_labels.append( [meningioma_bk[0][2]] * len(kneighb) )

            batch_features.append( get_image(glioma_bk[0][0],glioma_bk[0][1],kneighb) )
            batch_labels.append( [glioma_bk[0][2]] * len(kneighb)  )


            batch_features.append( get_image(pituitary_bk[0][0],pituitary_bk[0][1],kneighb) )
            batch_labels.append( [pituitary_bk[0][2]]* len(kneighb))
        
        last_batch.append(meningioma_bk[0])
        last_batch.append(glioma_bk[0])
        last_batch.append(pituitary_bk[0])
        
        del meningioma_bk[0]
        del glioma_bk[0]
        del pituitary_bk[0]
    
    #batch_labels_transformed = one_hot_encode(batch_labels)
        
    #return [np.array(batch_features), np.array(batch_labels_transformed)]
    return [np.array(batch_features), np.array(batch_labels)]
    
def display_last_batch():
    for i in range(len(last_batch)):
        temp_i = np.array(get_image(last_batch[i][0],last_batch[i][1],0),dtype=float)
        #plt.imshow(temp_i,title=str(last_batch[i][2]) )
        plt.imshow(temp_i)
        plt.show()
        time.sleep(2)
    
def get_tumor_neighbor_last_batch(i):
    batch_features = []
    batch_labels = []
    for j in range(len(last_batch)):
        batch_features.append( get_image(last_batch[j][0],last_batch[j][1],i) )
        #batch_labels.append( (last_batch[i][2]) )
        
    return np.array(batch_features)#, np.array(batch_labels)]
        

def get_validation(epoch_batch_size, epoch_valid_size):

    reload_input_data()
    
    min_bk_data_set = min(len(pituitary_bk), len(glioma_bk), len(meningioma_bk))
    test_bk_data_size = int(epoch_batch_size * epoch_valid_size / 100)
    
    return get_a_batch(test_bk_data_size,0)

def image_multiplier(batch_features, batch_labels, neighbor_list):
    feat_batch2 = np.zeros((len(neighbor_list), len(batch_features),512,512),dtype=float)
    label_batch2 = np.zeros((len(neighbor_list), len(batch_labels)),dtype=int)
    feat_batch_final = np.zeros(((len(neighbor_list)*4), len(batch_features),512,512,1),dtype=float)
    label_batch_final = np.zeros(((len(neighbor_list)*4), len(batch_labels)),dtype=int)
    for i in range(len(batch_features)):
        for j in range(len(neighbor_list)):
            feat_batch2[j][i] = batch_features[i][j]
            label_batch2[j][i] = batch_labels[i][j]

            for i in range(len(feat_batch2)):
                for j in range(4):
                    temp_np = np.rot90(feat_batch2[i],j,(1,2))
                    feat_batch_final[i+j] = (temp_np[:,:,:,np.newaxis] )
                    label_batch_final[i+j] = label_batch2[i]
    
    return [feat_batch_final, label_batch_final]

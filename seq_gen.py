import numpy as np
import os
#directory of the dataset
dir_train = '../WriterID/Data/Train'
dir_test = '../WriterID/Data/Validation'
#write directory
write_train = './train'
write_test = './test'
if not os.path.isdir(write_train):
    os.mkdir(write_train)
if not os.path.isdir(write_test):
    os.mkdir(write_test)

#length of the sequence
seq_length = 40
interval = 10
num_sample_train = 1000
num_sample_test = 500
#
filenames = os.listdir(dir_train)
numclass = len(filenames)
print('class number=', numclass)
#training
lastperson_idx = 0
idx = 0
for i in range(numclass):
    #characters belong to a person
    character_person = np.load(os.path.join(dir_train, filenames[i]))
    #a character
    for character_id in range(len(character_person)):
        # a character
        datapoints = np.zeros((1, 3))
        character = character_person[character_id]
 
        for stroke_id in range(len(character)):
            # a stroke
            stroke = character[stroke_id]
            stroke_len = len(stroke)
            for point_id in range(stroke_len):
                datapoints = np.append(datapoints, np.expand_dims(np.append(stroke[point_id], 1), axis = 0), axis=0)
            datapoints[-1, -1] = 0
        x_min = np.min(datapoints[1:, 0])
        x_max = np.max(datapoints[1:, 0])
        y_min = np.min(datapoints[1:, 1])
        y_max = np.max(datapoints[1:, 1])   
        #print('x_min is ', x_min)     
        if((x_max-x_min)>1e-2):
            datapoints[1:, 0] = (datapoints[1:, 0] - x_min)/(x_max - x_min)
        else: 
            datapoints[1:, 0] = (datapoints[1:, 0] - x_min)/x_max
        if((y_max-y_min)>1e-2):
            datapoints[1:, 1] = (datapoints[1:, 1] - y_min)/(y_max - y_min)
        else:
            datapoints[1:, 1] = (datapoints[1:, 1] - y_min)/y_max

        points_seq = np.array([np.append((datapoints[i+1]-datapoints[i])[0:2], datapoints[i, 2]) for i in range(1, len(datapoints)-1)])
        points_length = len(points_seq)

        while(points_length>=seq_length):
            # a sequence
            if(points_length == seq_length):
                save_seq = np.array(points_seq[-points_length:])
            else:
                save_seq = np.array(points_seq[-points_length:-points_length+seq_length])
            points_length = points_length - interval
            scalar_label = np.zeros((1,))
            scalar_label[0] = i
            instance = {}
            instance['data'] = save_seq
            instance['gt'] = scalar_label
            np.save(os.path.join(write_train, str(seq_length)+'_'+str(idx)+'.npy'), instance)
            idx = idx + 1
            if (idx-lastperson_idx)>num_sample_train:
                break
        if (idx-lastperson_idx)>num_sample_train:
            break
    print('the {} th person have {} training stroke sequences'.format(i+1, idx-lastperson_idx))
    lastperson_idx = idx
print('total stroke number: {}'.format(lastperson_idx))
#testing
lastperson_idx = 0
idx = 0
for i in range(numclass):
    #characters belong to a person
    character_person = np.load(os.path.join(dir_test, filenames[i]))
    #a character
    for character_id in range(len(character_person)):
        datapoints = np.zeros((1, 3))
        character = character_person[character_id]

        for stroke_id in range(len(character)):
            # a stroke
            stroke = character[stroke_id]
            stroke_len = len(stroke)
            for point_id in range(stroke_len):
                datapoints = np.append(datapoints, np.expand_dims(np.append(stroke[point_id], 1), axis = 0), axis=0)
            datapoints[-1, -1] = 0

        x_min = np.min(datapoints[1:, 0])
        x_max = np.max(datapoints[1:, 0])
        y_min = np.min(datapoints[1:, 1])
        y_max = np.max(datapoints[1:, 1])        
        datapoints[1:, 0] = (datapoints[1:, 0] - x_min)/(x_max - x_min)
        datapoints[1:, 1] = (datapoints[1:, 1] - y_min)/(y_max - y_min)


        points_seq = np.array([np.append((datapoints[i+1]-datapoints[i])[0:2], datapoints[i, 2]) for i in range(1, len(datapoints)-1)])
        points_length = len(points_seq)

        while(points_length>=seq_length):
            # a sequence
            if(points_length == seq_length):
                save_seq = np.array(points_seq[-points_length:])
            else:
                save_seq = np.array(points_seq[-points_length:-points_length+seq_length])
            points_length = points_length - interval
            scalar_label = np.zeros((1,))
            scalar_label[0] = i
            instance = {}
            instance['data'] = save_seq
            instance['gt'] = scalar_label
            np.save(os.path.join(write_test, str(seq_length)+'_'+str(idx)+'.npy'), instance)
            idx = idx + 1
        if (idx-lastperson_idx)>num_sample_test:
            break
    print('the {} th person have {} testing stroke sequences'.format(i+1, idx-lastperson_idx))
    lastperson_idx = idx
print('total stroke number: {}'.format(lastperson_idx))

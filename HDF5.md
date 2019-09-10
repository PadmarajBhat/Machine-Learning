##### I just came to know about the HDF5 file format, making notes as I go through the articles
* Has compression ration upto 50%
* easily readable/saved through panda
* does a delayed operation to save memory
* libraries are written in c (fast and portable)
* data stored in hierarchical structure (file and folders) and can have meta data associated at each level.
  *  dataset = equivalent to numpy array
  *  group = collection of files
  * file can have group or files
  
* for some reason people have moved away from this format as distributed as gained popularity. Closing the further investigation !!!
    * https://cyrille.rossant.net/moving-away-hdf5/

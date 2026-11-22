# AI Engineering Take-Home

### 1. Timings
my first version with k = 9 showed around 6 to 7 hours, then i switched to K = 5 and ran 2 batches of sizes 25 in parallel that halved the total time from 3 hours to approximately 1 hour and 20 mins, the final version took around 75 mins.

### 2. Pipeline
the script works in the following order: first of all prepare() is called with params which does necessary imports and then downloads yolo11l and checks dependencies then the .parquet input files are read and every second of every video is grabbed and k (5 in our case) frames are processed and yolo11l finds objects and then writes the output in .jsonl file using strict formatting.

### 3. Where the Time Goes
most of the time in final version was due to cpu had to convert color spaces and resize frames during which gpu sits idle, but on RTX of yours CPU's are powerful and GPU's wont starve and I measured this by using time.time() across workers and batchsizes.

### 4. Checking Correctness
I ran unit tests for edge cases so that jsonl schema isn't corrupted and when i made it faster (using k = 5) i checked its accuracy by using a small subset of videos and ran them with k = 30 so that baseline is set. then I compared that against my k = 5 model and F1 was 0.97 above the threshold so accuracy got verified.

### 5. Working with AI
the issue that wasted a lot of my time was passing 4d numpy array into model.predict() which throws error on kaggle so i rejected it and used 3d image list.

### 6. Next Steps
instead of multiprocessing queue, and flattening thousands of bytes to put in Numpy array we can use shared memory. we will fix slots of sizes in the memory and the main process will read directly from the RAM's reserved slot so GPU does not starve and CPU's processing is reduced.

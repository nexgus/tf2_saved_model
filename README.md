# A Simple Demostration: Add Custom Info to TensorFlow SavedModel
A TensorFlow SavedModel can add "custom information" in it. Here a simple example to demostrate how to add custom information in model.
-   Clone source code.
    ```bash
    cd ~
    git clone https://github.com/nexgus/tf2_saved_model
    ```
-   It is a good idea to use virtual environment.
    ```bash
    cd tf2_saved_model
    virtualenv -p python3.8 .
    source bin/activate
    ```
-   Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
-   Save an model with extra info. This info contains a number and a labelmap (as an asset).
    ```bash
    python save.py
    ```
    You can see a `saved_model` directory is created. Find the `saved_model/assets` directory. You will find a file `labelmap.txt` in it.
-   We can load number and labelmap after load the model.
    ```bash
    python load.py
    ```
    The output will be
    ```
    number: 0.12345000356435776
    asset: ./saved_model/assets/labelmap.txt
    labelmap: ['label 0', '中文標籤 1', 'label 2'])
    label 0: label 0
    label 1: 中文標籤 1
    label 2: label 2
    ```
-   Now let's saving model without labelmap.
    ```bash
    python save.py --no_labelmap
    ```
    You'll get a warning.
    ```
    WARNING:absl:Found untraced functions such as get_label, get_labelmap, get_labelmap_asset while saving (showing 3 of 3). These functions will not be directly callable after loading.
    ```
-   Let's trying to load it.
    ```bash
    python load.py
    ```
    Here the example output:
    ```
    number: 0.12345000356435776
    Traceback (most recent call last):
      File "load.py", line 13, in <module>
        asset = model.info.get_labelmap_asset().numpy()
      File "/home/nexgus/myproj/tf_saved_model/lib/python3.8/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
        raise e.with_traceback(filtered_tb) from None
      File "/home/nexgus/myproj/tf_saved_model/lib/python3.8/site-packages/tensorflow/python/saved_model/function_deserialization.py", line 263, in restored_function_body
        raise ValueError("Found zero restored functions for caller function.")
    ValueError: Found zero restored functions for caller function.
    ```
    You may see the `nummber` is retrived sucessfully. However, while we call `model.info.get_labelmap_asset()`, we got an `ValueError` exception.

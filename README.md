## Installation Rules
1. For `R` to be able to install packages in `/usr/local/lib/R/` in Ubuntu, add yourselves to the group called `staff`
    ```{sh}
    sudo usermod -a -G staff <your_user_name>
    ```

1.  To install `car` R package in ubuntu, you will first needf to install the following apt packages. Mor details in this [SO-Link](https://stackoverflow.com/a/35650554/1652217)
    ```{sh}
    sudo apt install liblapack-dev liblapack3 libopenblas-base libopenblas-dev
    ```
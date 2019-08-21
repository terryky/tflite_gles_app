# utilities for Raspberry Pi

#### How to build
Utilities in this directory depend on Raspberry Pi official OSS. 
You need to clone the source code. 
```
> cd ~/work/
> git clone https://github.com/raspberrypi/userland.git raspberrypi_userland
```

and then, add include path to the Makefile.

```
INCLUDES += $(HOME)/work/raspberrypi_userland/
```

#### Reference
source codes in this directory came from:
https://github.com/raspberrypi/userland/tree/master/host_applications/linux/apps/raspicam

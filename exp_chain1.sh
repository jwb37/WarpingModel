python3 train.py RTN_Shoes

if [ -d checkpoints/RTN_Shoes/Shoes ]; then rm -rf checkpoints/RTN_Shoes/Shoes; fi
python3 test.py RTN_Shoes FlowGen
python3 test.py RTN_Shoes WarpedGen
mv checkpoints/RTN_Shoes/Shoes 'checkpoints/RTN_Shoes/Shoes(no_blur)'

if [ -d checkpoints/RTN_Shoes/Shoes ]; then rm -rf checkpoints/RTN_Shoes/Shoes; fi
python3 test.py RTN_Shoes_Blur FlowGen
python3 test.py RTN_Shoes_Blur WarpedGen
mv checkpoints/RTN_Shoes/Shoes 'checkpoints/RTN_Shoes/Shoes(blur)'

python3 train.py Mimic_Shoes
python3 test.py Mimic_Shoes WarpedGen
python3 train.py Pix2Pix_Shoes
python3 test.py Pix2Pix_Shoes I2I

python3 train.py Mimic_Shoes_Blur
python3 test.py Mimic_Shoes_Blur WarpedGen
python3 train.py Pix2Pix_Shoes_Blur
python3 test.py Pix2Pix_Shoes_Blur I2I

python3 train.py Pix2Pix_Shoes_Base
python3 test.py Pix2Pix_Shoes_Base I2I
python3 train.py Pix2Pix_Shoes_WarpsOnly
python3 test.py Pix2Pix_Shoes_WarpsOnly I2I

python3 train.py Pix2Pix_Shoes_RTN
python3 test.py Pix2Pix_Shoes_RTN I2I
python3 train.py Pix2Pix_Shoes_RTN_Blur
python3 test.py Pix2Pix_Shoes_RTN_Blur

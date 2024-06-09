#python train_MEK.py --expt_name=MEK_CE_0 --noise_ratio=0 --constraints=0 --dataset='affectnet' --epochs=5 --seed=0 --batch_size=16 --lr=2.5e-4
#python train_EAC.py --expt_name=EAC_CE_0 --noise_ratio=0 --constraints=0 --dataset='affectnet' --epochs=5 --seed=0 --batch_size=128


export CUDA_VISIBLE_DEVICES=0

for i in 0 #1
do
	for j in 0 #10 20 30
	do

		## AFFWILD ##
		#Uncalibrated#
		python train_EAC.py --expt_name=EAC_CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --epochs=20 --seed=$i --batch_size=128 --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'

		#Calibrated#
		python train_EAC.py --expt_name=EAC_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --epochs=20 --seed=$i --batch_size=128 --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'


		## RAFDB ##
		#Uncalibrated#
		python train_EAC.py --expt_name=EAC_CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --batch_size=128 --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'

		#Calibrated#
		python train_EAC.py --expt_name=EAC_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --batch_size=128 --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'


		## AFFECTNET ##
		#Uncalibrated#
		python train_EAC.py --expt_name=EAC_CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i --batch_size=128

		#Calibrated#
		python train_EAC.py --expt_name=EAC_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed=$i --batch_size=128

		echo $j
	done

done

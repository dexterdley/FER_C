export CUDA_VISIBLE_DEVICES=0

for i in 0 1 2
do
	for j in 0 10 20 30
	do

		## AFFWILD ##
		#Uncalibrated#
		python train_soft_calibration.py --expt_name=CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_SCN.py --expt_name=SCN_CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --relabel_epoch=15 --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_DMUE.py --expt_name=DMUE_CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_RUL.py --expt_name=RUL_CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_SWINV2.py --expt_name=SWINV2_CE_$j --noise_ratio=$j --constraints=0 --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'

		#Calibrated#
		python train_soft_calibration.py --expt_name=maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_SCN.py --expt_name=SCN_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --relabel_epoch=15 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_DMUE.py --expt_name=DMUE_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_RUL.py --expt_name=RUL_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'
		python train_SWINV2.py --expt_name=SWINV2_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affwild' --epochs=20 --seed=$i --file_path='/media/dex/80F46FA8F46F9EE2/Users/Admin/Desktop/NUS/CS6101/Affwild/images/'

		## RAFDB ##
		#Uncalibrated#
		python train_soft_calibration.py --expt_name=CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_SCN.py --expt_name=SCN_CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_DMUE.py --expt_name=DMUE_CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_RUL.py --expt_name=RUL_CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_SWINV2.py --expt_name=SWINV2_CE_$j --noise_ratio=$j --constraints=0 --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'

		#Calibrated#
		python train_soft_calibration.py --expt_name=maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_SCN.py --expt_name=SCN_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_DMUE.py --expt_name=DMUE_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_RUL.py --expt_name=RUL_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'
		python train_SWINV2.py --expt_name=SWINV2_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='rafdb' --seed=$i --file_path='/home/dex/Desktop/RC_AffectNet/RAF-DB/aligned/'


		## AFFECTNET ##
		#Uncalibrated#
		python train_soft_calibration.py --expt_name=CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i
		python train_SCN.py --expt_name=SCN_CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i
		python train_DMUE.py --expt_name=DMUE_CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i
		python train_RUL.py --expt_name=RUL_CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i
		python train_SWINV2.py --expt_name=SWINV2_CE_$j --noise_ratio=$j --constraints=0 --dataset='affectnet' --epochs=5 --seed=$i

		#Calibrated#
		python train_soft_calibration.py --expt_name=maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed=$i
		python train_SCN.py --expt_name=SCN_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed=$i
		python train_DMUE.py --expt_name=DMUE_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed=$i
		python train_RUL.py --expt_name=RUL_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed$i
		python train_SWINV2.py --expt_name=SWINV2_maxent_mu_mbls_$j --noise_ratio=$j --constraints=6 --mbls=True --dataset='affectnet' --epochs=5 --seed=$i

		echo $j
	done

done

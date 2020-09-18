#/bin/bash
#$ -S /bin/bash
set -x -e

GREEDY_HOME="/home/sadhana-ravikumar/Documents/packages/greedy/bin"
ROOT='/home/sadhana-ravikumar/Documents/Sadhana/unet3d_srlm'
INPUTS=$ROOT/inputs
DATADIR=$ROOT/data_csv
CODEDIR=$ROOT/code

SUBJ_TXT=$ROOT/subj_train.txt
IND_ALL="$(cat $SUBJ_TXT)"
TEMPLATE="/home/sadhana-ravikumar/Documents/Sadhana/STN_pytorch3d/template_2016/template_crop_hf_srlm.nii.gz"

function main()
{

  mkdir -p $DATADIR
#  process_inputs

  Preparation
 PreparationTest
}

function process_inputs()
{

  FILE=$ROOT/subj_test.txt
  N=$(cat $FILE | wc -l)

  for ((i=1;i<=${N};i++)); do

        LINE=$(cat $FILE | head -n $i | tail -n 1)
        id=$(echo $LINE | cut -d ' ' -f 1)

	echo $id

	#Get images
	HF_SEG=${INPUTS}/${id}/${id}_hfseg.nii.gz
	SRLM_SEG=${INPUTS}/${id}/${id}_dbseg.nii.gz
	IMG=${INPUTS}/${id}/${id}_img_n4.nii.gz

	#Combine segmentations into single multilabel
	SEG=$INPUTS/${id}/${id}_multilabel_seg.nii.gz	
	#c3d $HF_SEG $SRLM_SEG -add -o $SEG

	#echo "Normalizing image for  $id"
	IMG=$INPUTS/${id}/${id}_img_n4.nii.gz
	SEG=$INPUTS/${id}/${id}_multilabel_seg.nii.gz

 	IMG_TRIM=$INPUTS/${id}/${id}_trimmed_img.nii.gz
	SEG_TRIM=$INPUTS/${id}/${id}_trimmed_phg.nii.gz

<<"SKIP"
      #Trim input image to only contain segmented region
	c3d $SEG -trim 15vox -o $SEG_TRIM -thresh -inf inf 1 0 -popas MASK $IMG \
	-push MASK -reslice-identity -as R $IMG -add -push R -times -trim 0vox \
        -shift -1 -o $IMG_TRIM

SKIP
       # Register with MOI, the template to img

	$GREEDY_HOME/greedy -d 3 -i $SEG_TRIM $TEMPLATE -moments -m NMI -o $INPUTS/${id}/${id}_moments.mat
	
	# Try affine registration
        $GREEDY_HOME/greedy -d 3 -a -i $SEG_TRIM $TEMPLATE -ia $INPUTS/${id}/${id}_moments.mat \
	-m NMI -n 100x50x0 -o $INPUTS/${id}/${id}_affine.mat

        $GREEDY_HOME/greedy -d 3 -rf $IMG_TRIM -ri LABEL 0.24vox \
	 -rm $TEMPLATE $INPUTS/${id}/${id}_template_prior_resliced_affine.nii.gz \
	-r $INPUTS/${id}/${id}_affine.mat
  
	#Downsample the trimmed input image since such a high res is not required. Patch will capture more info
  #	c3d $IMG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_img.nii.gz
  #	c3d $SEG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_phg.nii.gz
	# Post processing to visualize test results

  done
}

function Preparation()
{ 

  N=$(cat $SUBJ_TXT | wc -l)
  rm -rf $DATADIR/phg_split.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $SUBJ_TXT | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $SUBJ_TXT | grep $id)   
  
    IMG=$INPUTS/${id}/${id}_trimmed_img.nii.gz
    SEG=$INPUTS/${id}/${id}_trimmed_phg.nii.gz
    TEMPLATE=$INPUTS/${id}/${id}_template_prior_resliced_affine.nii.gz
    echo "$IMG,$SEG,$idint,"Control",$type",$TEMPLATE >> $DATADIR/phg_split.csv

  done
}

function PreparationTest()
{

  N=$(cat $ROOT/subj_test.txt | wc -l)
  rm -rf $DATADIR/phg_split_test.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $ROOT/subj_test.txt | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $ROOT/subj_test.txt | grep $id)

      IMG=$INPUTS/${id}/${id}_trimmed_img.nii.gz
      SEG=$INPUTS/${id}/${id}_trimmed_phg.nii.gz
      TEMPLATE=$INPUTS/${id}/${id}_template_prior_resliced_affine.nii.gz
      echo "$IMG,$SEG,$idint,"Control",$type",$TEMPLATE >> $DATADIR/phg_split_test.csv

  done
}
main

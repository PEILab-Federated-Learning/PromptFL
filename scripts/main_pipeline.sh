#!/bin/bash

cd ..

# custom config
DATA="your dataset path"
TRAINER=PrompFL
PRETRAINED=True
LR=0.001

#DATASET=$1
CFG=$1  # config file
CTP=$2  # class token position (end or middle)
NCTX=$3  # number of context tokens
IID=$4
CSC=$5  # class-specific context (False or True)
USEALL=$6
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in caltech101
do
  for SHOTS in 1
  do
    for REPEATRATE in 0.0
    do
      for USERS in 64
      do
        for EPOCH in 5
        do
          for ROUND in 20
          do
            for SEED in 1
            do
              DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/pretrain_${PRETRAINED}/iid_${IID}_repeatrate_${REPEATRATE}/${USERS}_users/lr_${LR}/${EPOCH}epoch_${ROUND}round/seed${SEED}
              if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
              else
                python federated_main.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                TRAINER.PROMPTFL.N_CTX ${NCTX} \
                TRAINER.PROMPTFL.CSC ${CSC} \
                TRAINER.PROMPTFL.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.USERS ${USERS} \
                DATASET.IID ${IID} \
                DATASET.REPEATRATE ${REPEATRATE} \
                OPTIM.MAX_EPOCH ${EPOCH} \
                OPTIM.ROUND ${ROUND}\
                OPTIM.LR ${LR}\
                MODEL.BACKBONE.PRETRAINED ${PRETRAINED}\
                DATASET.USEALL ${USEALL}
              fi
            done
          done
        done
      done
    done
  done
done


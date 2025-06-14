#!/bin/bash

TEACHER="Pretrain/ml-100k_NeuMF_32_[64,32,16,8]_1749891631.npy"
TECHNIQUE="response"  # change to: response, feature, mlp_student, relation
LAYERS="[8]"     # you can change it here
REG_LAYERS="[0]"
FACTORS=8              # embedding size for student
LOGFILE="logs${LAYERS}/${TECHNIQUE}.txt"

mkdir -p logs${LAYERS}

#response
for i in {1..10}
do
  echo "Run $i" | tee -a $LOGFILE
  python ex_12.py \
    --teacher_model "$TEACHER" \
    --technique "$TECHNIQUE" \
    --layers "$LAYERS" \
    --num_factors "$FACTORS" \
    --reg_layers "$REG_LAYERS" \
    --epochs 11 \
    --out 0 \
    | tee -a $LOGFILE
done

#feature
TECHNIQUE="feature"
LOGFILE="logs${LAYERS}/${TECHNIQUE}.txt"
for i in {1..10}
do
  echo "Run $i" | tee -a $LOGFILE
  python ex_12.py \
    --reg_layers "$REG_LAYERS" \
    --teacher_model "$TEACHER" \
    --technique "$TECHNIQUE" \
    --layers "$LAYERS" \
    --num_factors "$FACTORS" \
    --epochs 11 \
    --out 0 \
    | tee -a $LOGFILE
done

#student_mlp
TECHNIQUE="relation"
LOGFILE="logs${LAYERS}/${TECHNIQUE}.txt"
for i in {1..10}
do
  echo "Run $i" | tee -a $LOGFILE
  python ex_12.py \
    --reg_layers "$REG_LAYERS" \
    --teacher_model "$TEACHER" \
    --technique "$TECHNIQUE" \
    --layers "$LAYERS" \
    --num_factors "$FACTORS" \
    --epochs 11 \
    --out 0 \
    | tee -a $LOGFILE
done
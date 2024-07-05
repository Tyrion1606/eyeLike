#!/bin/bash

# Define a mensagem de commit padrão
commit_text="."

# Se um parâmetro for passado, use-o como a mensagem de commit
if [ ! -z "$1" ]; then
  commit_text=$1
fi

# Executa os comandos do Git
git add .
git commit -m "$commit_text"
git push origin dialing
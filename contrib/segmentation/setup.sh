#!/bin/bash
WORKSPACE="workshop"
RESOURCE_GROUP="semantic-segmentation-workshop"
LOCATION="westus2"

# Setup Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az upgrade

# Install Azure ML CLI v2
# NOTE: We are also removing the v1 CLI with this step
az extension remove -n azure-cli-ml
az extension remove -n ml
az extension add -n ml -y

# Setup AML Workspace
az group create -n $RESOURCE_GROUP -l $LOCATION
az ml workspace create -n $WORKSPACE -g $RESOURCE_GROUP -l $LOCATION
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 4 -d 300 --size Standard_DS3_V2
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 4 -d 300 --size Standard_NC6

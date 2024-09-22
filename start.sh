#!/bin/bash


INSTANCE_IP=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=boc" --query "Reservations[*].Instances[*].PublicIpAddress" --output text)

# Check if the IP address was fetched successfully
if [ -z "$INSTANCE_IP" ]; then
  echo "No instance found with the tag 'Name=boc' or instance does not have a public IP address."
  exit 1
fi

echo "Fetched IP address: $INSTANCE_IP"

# Use the fetched IP address in the scp commands

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no /Users/plato/.ssh/id_remotekey ubuntu@$INSTANCE_IP:~/.ssh/id_rsa
if [ $? -ne 0 ]; then
    echo "unable to connect"
    exit 1
fi

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no /Users/plato/.ssh/id_remotekey.pub ubuntu@$INSTANCE_IP:~/.ssh/id_rsa.pub

ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'chmod 600 ~/.ssh/id_rsa'
echo "Changed permissions for id_rsa on the EC2 instance."

ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'ssh-keyscan -H github.com >> ~/.ssh/known_hosts'
ssh -i ~/.ssh/id_sache ubuntu@$INSTANCE_IP 'git clone git@github.com:lewington-pitsos/boc'

scp -i ~/.ssh/id_sache -o StrictHostKeyChecking=no .credentials.json ubuntu@$INSTANCE_IP:~/boc/
echo "Copied credentials over"

sed -i '' "/^Host aws-boc$/,/^$/ s/HostName .*/HostName $INSTANCE_IP/" /Users/plato/.ssh/config
#!/bin/bash
if [ -z "$API_KEY" ]; then
    echo "Need to export API_KEY"
    exit
fi

curl -X POST -u "apikey:${API_KEY}" \
    --header "Content-Type: audio/wav" \
    --data-binary @sounds/"$1" \
    https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/7311c985-e03f-4d80-80d5-fcace75363b9/v1/recognize?model=es-ES_BroadbandModel

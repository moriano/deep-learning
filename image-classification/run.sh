#!/bin/bash
echo "Are you sure you have downloaded the latest results? yes/no"
read answer
if [ "$answer" = "yes" ]; then
	echo "Do we go with cpu or gpu?"
	read type 
	floyd run --$type --env tensorflow --mode jupyter --data diSgciLH4WA7HpcHNasP9j
fi


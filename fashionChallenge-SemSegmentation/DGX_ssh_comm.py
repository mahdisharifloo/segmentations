# Sync files with DGX workstation via SSH and exec cmds
# Uses putty commands (windows)
#
# Eduardo January 2019

import os
import argparse
import codecs

# TODO: implement bash file transfer and execution (-f)
# TODO: implement for linux/mac 
# TODO: add args for username, password, host, port

"""
######################## 
# Usage
########################

'-c <cmd>' : execute command  
'-s <src> -d <dest>' : sync folder
    copy (override) all files in folder
    <src>: source on local
    <dest>: destnation on remote (if None, dest = temp folder in Documents)

Example - sync and execute train.py:
python DGX_SSH_comm.py -c "python train.py" -s /Users/T0R17FL/Documents/fashionChallenge-SemSegmentation -d /home/eduardo/Documents/fashionChallenge-SemSegmentation/
python DGX_SSH_comm.py -s /Users/T0R17FL/Documents/fashionChallenge-SemSegmentation -d /home/eduardo/Documents/fashionChallenge-SemSegmentation/

NOTE: sync is always done before executing commands
NOTE: password is read from file "password"
NOTE: dot-starting files are ignored

open connection with port for tensorboard
ssh -L 16006:127.0.0.1:6006 eduardo@scott-machineauto.jdnet.deere.com
"""

# read args
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--syncsrc", required=False, help="sync src")
ap.add_argument("-d", "--syncdest", required=False, help="sync dest")
ap.add_argument("-c", "--command", required=False, help="command")

args = vars(ap.parse_args())
src = args["syncsrc"]
dest = args["syncdest"]
cmd = args["command"]

username = "eduardo"
host = "scott-machineauto.jdnet.deere.com"
remote = username + "@" + host
with codecs.open('.pwrd', 'r', 'utf-16') as file:
    password=file.read().rstrip("\r\n")


# if src defined: sync
if src is not None:
    # default value for dest
    if dest is None:
        temp_rm = " \"rm -r /home/eduardo/Documents/temp_syncDGX\""
        temp = " \"mkdir /home/eduardo/Documents/temp_syncDGX\""
        os.system("plink -pw odr18edu " + remote + temp_rm)
        os.system("plink -pw odr18edu " + remote + temp)
        dest = "/home/eduardo/Documents/temp"
    file_names = [os.path.join(src, f) for f in os.listdir(src) 
        if os.path.isfile(os.path.join(src, f)) and not f[0] =="."] 
    for file_name in file_names:
        command = "pscp -pw " + password + " " + file_name + " " + remote + ":" + dest
        os.system(command)

# if command defined
if cmd is not None:
    os.system("plink -pw " + password + " " + remote + " \""+ cmd +"\"")



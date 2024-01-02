import paramiko
import csv

def append_to_remote_csv(hostname, port, username, password, remote_csv_path, new_commands):
    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        ssh.connect(hostname, port, username, password)

        # Read the existing content of the remote CSV file
        with ssh.open_sftp() as sftp:
            with sftp.file(remote_csv_path, 'r') as remote_csv_file:
                first_line = remote_csv_file.readline().strip()

            # Append new commands to the CSV file
            with sftp.file(remote_csv_path, 'a') as remote_csv_file:
                writer = csv.writer(remote_csv_file)
                for command in new_commands:
                    writer.writerow(command)

        print("Commands successfully added to the remote CSV file.")

        # Check the condition in the first line
        if first_line.lower() == 'false':
           command = "python3 /home/aviran/running/task.py"
           ssh.exec_command(command)
    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the SSH connection
        ssh.close()

 

if __name__ == '__main__':
    # Set the connection parameters
    hostname = '132.72.53.146'
    port = 22  # Default SSH port
    username = 'aviran'
    password = 'qwerty'
    remote_csv_path = '/home/aviran/running/tasks.csv'
    new_commands = [
        # write the path of the project, the name of the environment, the name of the python file and your email
        ['/home/aviran/Alon/Thesis', 'alon_env', 'ViTMLPRegressor.py', 'alonkay@post.bgu.ac.il']
    ]
    append_to_remote_csv(hostname, port, username, password, remote_csv_path, new_commands)

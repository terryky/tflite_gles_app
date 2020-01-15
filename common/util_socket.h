#ifndef _UTIL_SOCKET_H_
#define _UTIL_SOCKET_H_

int create_server_socket ();
int send_fd_to_server (int stream_fd);
int receive_fd_from_client (int socket);

int connect_to_server ();
int commit_fd_to_server (int socket, int stream_fd);

int connect_to_client (int socket);
int acquire_fd_from_client (int fd);

int open_server_socket (int port_num);
int open_client_socket (char *server_name, int port_num);

#endif /* _UTIL_SOCKET_H_ */


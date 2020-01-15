/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2019 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <netdb.h>
#include <arpa/inet.h>


#define TMP_SAMPLE_SOCK   "/tmp/sample_sock"

int
create_server_socket ()
{
    struct sockaddr_un un;
    int s, ret;

    s = socket (PF_UNIX, SOCK_STREAM, 0);
    if (s < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    unlink (TMP_SAMPLE_SOCK);

    memset (&un, 0, sizeof (un));
    un.sun_family = AF_UNIX;
    strcpy (un.sun_path, TMP_SAMPLE_SOCK);
    
    ret = bind (s, (struct sockaddr *)&un, sizeof (un));
    if (ret < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    listen (s, 3);

    return s;
}

int
connect_to_server ()
{
    struct sockaddr_un un;
    int s, ret;
    
    s = socket (PF_UNIX, SOCK_STREAM, 0);
    if (s < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    memset (&un, 0, sizeof (un));
    un.sun_family = AF_UNIX;
    strcpy (un.sun_path, TMP_SAMPLE_SOCK);
    
    ret = connect (s, (struct sockaddr *)&un, sizeof(un));
    if (ret < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return s;
}

int 
connect_to_client (int socket)
{
    struct sockaddr_un conn_addr = {0};
    socklen_t conn_addr_len = sizeof (conn_addr);
    int fd;
    
    fd = accept (socket, (struct sockaddr *)&conn_addr, &conn_addr_len);
    if (fd == -1)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    return fd;
}

int
send_fd_to_server (int stream_fd)
{
    int socket;
    
    socket = connect_to_server ();
    if (socket < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    {
        struct msghdr msg = {0};
        struct cmsghdr *cmsg;
        struct iovec iov;
        union {
            char buf[CMSG_SPACE (sizeof(int))];
            long align;
        } ctl;
        
        msg.msg_iov     = &iov;
        msg.msg_iovlen  = 1;
        iov.iov_base    = "x";
        iov.iov_len     = 1;
        msg.msg_control = ctl.buf;
        msg.msg_controllen = sizeof(ctl.buf);
        cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type  = SCM_RIGHTS;
        cmsg->cmsg_len   = CMSG_LEN(sizeof(int));

        memcpy(CMSG_DATA(cmsg), &stream_fd, sizeof(int));
        msg.msg_controllen = cmsg->cmsg_len;

        if (sendmsg(socket, &msg, 0) <= 0) 
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            goto fail;
        }
    }
    
    return 0;
    
fail:
    close (socket);
    return -1;
}

int
commit_fd_to_server (int socket, int stream_fd)
{
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    struct iovec iov;
    union {
        char buf[CMSG_SPACE (sizeof(int))];
        long align;
    } ctl;
    int ret;
    
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    iov.iov_base    = "x";
    iov.iov_len     = 1;
    msg.msg_control = ctl.buf;
    msg.msg_controllen = sizeof(ctl.buf);
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(int));

    memcpy(CMSG_DATA(cmsg), &stream_fd, sizeof(int));
    msg.msg_controllen = cmsg->cmsg_len;

    ret = sendmsg(socket, &msg, 0);
    if (ret <= 0) 
    {
        fprintf (stderr, "ERR: %s(%d): ret(%d)\n", __FILE__, __LINE__, ret);
        perror("sendmsg()");
        return -1;
    }
    return 0;
}


int
acquire_fd_from_client (int fd)
{
    int stream_fd;
    struct msghdr msg = {0};
    struct iovec  iov;
    struct cmsghdr *cmsg;
    char msg_buf[1];
    char ctl_buf[CMSG_SPACE (sizeof (int))];

    iov.iov_base    = msg_buf;
    iov.iov_len     = sizeof (msg_buf);
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    msg.msg_control = ctl_buf;
    msg.msg_controllen = sizeof (ctl_buf);

    if (recvmsg (fd, &msg, MSG_DONTWAIT) < 0)
    {
        //fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    cmsg = CMSG_FIRSTHDR (&msg);
    if ((cmsg == NULL )                  ||
        (cmsg->cmsg_level != SOL_SOCKET) ||
        (cmsg->cmsg_type  != SCM_RIGHTS))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -2;
    }

    memcpy (&stream_fd, CMSG_DATA (cmsg), sizeof (int));

    return stream_fd;
}

int
receive_fd_from_client (int socket)
{
    int stream_fd;
    struct msghdr msg = {0};
    struct iovec  iov;
    struct cmsghdr *cmsg;
    char msg_buf[1];
    char ctl_buf[CMSG_SPACE (sizeof (int))];

    struct sockaddr_un conn_addr = {0};
    socklen_t conn_addr_len = sizeof (conn_addr);
    int fd;
    
    fd = accept (socket, (struct sockaddr *)&conn_addr, &conn_addr_len);
    if (fd == -1)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    iov.iov_base    = msg_buf;
    iov.iov_len     = sizeof (msg_buf);
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    msg.msg_control = ctl_buf;
    msg.msg_controllen = sizeof (ctl_buf);

    if (recvmsg (fd, &msg, 0) <= 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    cmsg = CMSG_FIRSTHDR (&msg);
    if ((cmsg == NULL )                  ||
        (cmsg->cmsg_level != SOL_SOCKET) ||
        (cmsg->cmsg_type  != SCM_RIGHTS))
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    memcpy (&stream_fd, CMSG_DATA (cmsg), sizeof (int));

    return stream_fd;
}




int
open_server_socket (int port_num)
{
    const int on = 1;
    int  fdsock, server_len, ret;
    struct sockaddr_in server_addr;

    fdsock = socket (AF_INET, SOCK_STREAM, 0);
    if (fdsock < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    setsockopt (fdsock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof (on));
    setsockopt (fdsock, SOL_SOCKET, SO_REUSEPORT, &on, sizeof (on));

    memset (&server_addr, 0, sizeof (server_addr));
    server_addr.sin_family      = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port        = htons (port_num);
    server_len = sizeof (server_addr);

    ret = bind (fdsock, (struct sockaddr *)&server_addr, server_len);
    if (ret == -1)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }
    
    ret = listen (fdsock, 5);
    if (ret == -1)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    return fdsock;
}

int
open_client_socket (char *server_name, int port_num)
{
    int fdsock, ret, herr;
    struct sockaddr_in client_addr;
    struct hostent he, *hp;
    char tmpbuf[256];


    fdsock = socket (AF_INET, SOCK_STREAM, 0);
    if (fdsock < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    memset (&client_addr, 0, sizeof (client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_port   = htons (port_num);

    ret = gethostbyname_r (server_name, &he, tmpbuf, sizeof(tmpbuf), &hp, &herr);
    if (ret != 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        close (fdsock);
        return -1;
    }

    memcpy (&client_addr.sin_addr, hp->h_addr, hp->h_length);

    ret = connect (fdsock, (struct sockaddr *)&client_addr, sizeof (client_addr));
    if (ret < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        close (fdsock);
        return -1;
    }
    
    return fdsock;
}


/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>  // fork
#include <unistd.h>     // fork
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "util_socket.h"

#define PORT_NUM 12345

int
launch_client_app ()
{
    pid_t pid = fork ();

    if (pid == -1)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
    }
    else if (pid == 0)
    {
        // child
        execl("/usr/bin/python", "python", "client_sock.py", NULL);
    }
    else
    {
        // parent
    }
    return 0;
}



int
main(int argc, char *argv[])
{
    int fdsock = open_server_socket (PORT_NUM);
    if (fdsock < 0)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return 0;
    }

    launch_client_app ();

    while (1)
    {
        int client_sock;
        struct sockaddr_in client_addr;
        socklen_t len = sizeof (client_addr);
        char strACK[] = "ACK";

        client_sock = accept (fdsock, (struct sockaddr *)&client_addr, &len);
        if (client_sock < 0)
        {
            fprintf (stderr, "can't accept. retry.\n");
            continue;
        }

        fprintf (stderr, "connected from: %s\n", inet_ntoa (client_addr.sin_addr));

        while (1)
        {
            char buf[256];
            int num_rcv = recv (client_sock, buf, 256, 0);
            if (num_rcv < 0)
            {
                fprintf (stderr, "[S] num_rcv = %d\n", num_rcv);
                close (client_sock);
                break;
            }
            if (num_rcv == 0)
            {
                fprintf (stderr, "[S] connecion closed.\n");
                close (client_sock);
                break;
            }
            fprintf (stderr, "[S] RCVMSG[%d]: %s\n", num_rcv, buf);
            
            
            if (strcmp (buf, "HELLO") == 0 )
            {
                if (send (client_sock, strACK, strlen(strACK) + 1, 0) == -1)
                    fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            }
            else
            {
                if (send (client_sock, strACK, strlen(strACK) + 1, 0) == -1)
                    fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            }
        }
    }

    return 0;
}

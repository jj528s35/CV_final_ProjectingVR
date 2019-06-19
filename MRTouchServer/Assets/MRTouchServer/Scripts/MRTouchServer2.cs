using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MRTouchServer2 : MonoBehaviour {

    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;

	// Use this for initialization
	void Start () {
        tcpListenerThread = new Thread(new ThreadStart(ListenForIncommingRequests));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();
	}
	
	// Update is called once per frame
	void Update () {
		if(Input.GetKeyDown(KeyCode.Space))
        {
            SendMessage("Hi, this is server!");
        }
	}

    private void ListenForIncommingRequests()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Parse("127.0.0.1"), 7777);
            tcpListener.Start();
            Debug.Log("Server is listening");
            Byte[] bytes = new Byte[1024];
            while(true)
            {
                using (connectedTcpClient = tcpListener.AcceptTcpClient())
                {
                    using (NetworkStream stram = connectedTcpClient.GetStream())
                    {
                        int length;
                        while((length = stram.Read(bytes, 0, bytes.Length)) != 0)
                        {
                            var incommingData = new byte[length];
                            Array.Copy(bytes, 0, incommingData, 0, length);
                            string clietMessage = Encoding.ASCII.GetString(incommingData);
                            Debug.Log("client message received as: " + clietMessage);
                        }
                    }
                }
            }
        }catch(SocketException e)
        {
            Debug.Log("Socket exception " + e.ToString());
        }
    }

    private void SendMessage(string msg)
    {
        if(connectedTcpClient == null)
        {
            return;
        }

        try
        {
            NetworkStream stream = connectedTcpClient.GetStream();
            if (stream.CanWrite)
            {
                byte[] serverMessageAsByteArray = Encoding.ASCII.GetBytes(msg);
                stream.Write(serverMessageAsByteArray, 0, serverMessageAsByteArray.Length);
                Debug.Log("Server sent his message - should be received by client");
            }
        }catch(SocketException e)
        {
            Debug.Log(e.ToString());
        }
    }
}

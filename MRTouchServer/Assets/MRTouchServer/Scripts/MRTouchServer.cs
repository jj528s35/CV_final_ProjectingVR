using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class StateObject
{
    public Socket workSocket = null;
    public const int Buffersize = 1024;
    public byte[] buffer = new byte[Buffersize];
    public StringBuilder sb = new StringBuilder();
}

public class MRTouchServer : MonoBehaviour {

    public static ManualResetEvent allDone = new ManualResetEvent(false);

	// Use this for initialization
	void Start () {
        Debug.Log("start listening");
        StartListening();
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public static void StartListening()
    {
        IPAddress ipAddress = IPAddress.Parse("127.0.0.1");
        IPEndPoint localEndPoint = new IPEndPoint(ipAddress, 7777);
        Socket listener = new Socket(ipAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp);

        try
        {
            listener.Bind(localEndPoint);
            listener.Listen(3);

            //while (true)
            //{
                allDone.Reset();
                Debug.Log("Waiting for a connection...");
                listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
                allDone.WaitOne();
            //}
        }catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }

    public static void AcceptCallback(IAsyncResult ar)
    {
        allDone.Set();

        Socket listener = (Socket)ar.AsyncState;
        Socket handler = listener.EndAccept(ar);

        StateObject state = new StateObject();
        state.workSocket = handler;
        handler.BeginReceive(state.buffer, 0, StateObject.Buffersize, 0, new AsyncCallback(REadCallback), state);
    }

    public static void REadCallback(IAsyncResult ar)
    {
        String content = String.Empty;

        StateObject state = (StateObject)ar.AsyncState;
        Socket handler = state.workSocket;

        int bytesRead = handler.EndReceive(ar);

        if(bytesRead > 0)
        {
            state.sb.Append(Encoding.ASCII.GetString(state.buffer, 0, bytesRead));
            content = state.sb.ToString();
            if(content.IndexOf("<EOF>") > -1)
            {
                Debug.LogFormat("Read {0} bytes from socket.\n Data: {1}", content.Length, content);
                Send(handler, content);
            }
            else
            {
                handler.BeginReceive(state.buffer, 0, StateObject.Buffersize, 0, new AsyncCallback(REadCallback), state);
            }
        }
    }

    private static void Send(Socket handler, String data)
    {
        byte[] byteData = Encoding.ASCII.GetBytes(data);

        handler.BeginSend(byteData, 0, byteData.Length, 0, new AsyncCallback(SendCallback), handler);
    }

    private static void SendCallback(IAsyncResult ar)
    {
        try
        {
            Socket handler = (Socket)ar.AsyncState;
            int byteSend = handler.EndSend(ar);
            Debug.LogFormat("Sent {0} bytes to client.", byteSend);

            handler.Shutdown(SocketShutdown.Both);
            handler.Close();
        }catch(Exception e)
        {
            Debug.Log(e.ToString());
        }
    }
}

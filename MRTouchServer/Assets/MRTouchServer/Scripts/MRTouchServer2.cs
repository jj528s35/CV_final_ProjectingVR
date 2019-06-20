using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MRTouchServer2 : MonoBehaviour {

    public enum ReceiveType
    {
        None,
        fingertipsPos,
        touchedState
    };

    [Header("fingertips pos")]
    public int fingerNumber = 10;
    public List<Vector2> fingertips = new List<Vector2>();
    public List<bool> touchedState = new List<bool>();

    [Header("Debug")]
    public string debugFingertipsPos;
    public string debugTouchedState;

    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;

	// Use this for initialization
	void Start () {
        tcpListenerThread = new Thread(new ThreadStart(ListenForIncommingRequests));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();

        if(fingertips.Count != fingerNumber)
        {
            fingertips.Clear();
            for(int i=0;i<fingerNumber;i++)
            {
                fingertips.Add(Vector2.zero);
            }
        }

        if(touchedState.Count != fingerNumber)
        {
            touchedState.Clear();
            for(int i=0;i< fingerNumber;i++)
            {
                touchedState.Add(false);
            }
        }
	}

    void OnDiaable()
    {
        tcpListenerThread.Abort();
        tcpListenerThread.Join();
    }
	
	// Update is called once per frame
	void Update () {
		if(Input.GetKeyDown(KeyCode.Space))
        {
            SendMessage("Hi, this is server!");
        }

        if(Input.GetKeyDown(KeyCode.Alpha1))
        {
            ParseData(debugFingertipsPos);
            debugFingertipsPos = "";
        }

        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            ParseData(debugTouchedState);
            debugTouchedState = "";
        }

        if (Input.GetKeyDown(KeyCode.D))
        {
            ParseData(debugFingertipsPos);
            ParseData(debugTouchedState);
            debugFingertipsPos = "";
            debugTouchedState = "";
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
                        string clietMessage = "";
                        while ((length = stram.Read(bytes, 0, bytes.Length)) != 0)
                        {
                            var incommingData = new byte[length];
                            Array.Copy(bytes, 0, incommingData, 0, length);
                            clietMessage = Encoding.ASCII.GetString(incommingData);
                            ParseData(clietMessage);
                            Debug.Log("client message received as: " + clietMessage);
                        }
                    }
                }
            }
        }catch(SocketException e)
        {
            Debug.Log("Socket exception " + e.ToString());
        }catch(ThreadAbortException abortException)
        {
            Debug.Log(abortException);
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

    private void ParseData(string data)
    {
        string[] values = data.Split(' ');
        int dataType = int.Parse(values[0]);
        if(dataType == (int)ReceiveType.fingertipsPos)
        {
            if (values.Length-2 == (fingerNumber * 2))
            {
                for (int i = 0; i < fingerNumber; i++)
                {
                    Vector2 tip = fingertips[i];
                    if(tip == null)
                    {
                        tip = new Vector2();
                    }
                    tip.x = int.Parse(values[2*i + 1]);
                    tip.y = int.Parse(values[2*i + 1 + 1]);
                    fingertips[i] = tip;
                }
            }
            else
            {
                Debug.LogFormat("reveice fingertips positions, data length: {0}", values.Length);
                Debug.LogFormat("fingertips pos format is wrong: {0}", data);
            }
        }else if(dataType == (int) ReceiveType.touchedState)
        {
            //Debug.LogFormat("reveice touch state, data length: {0}", values.Length);
            if ((values.Length-2) == fingerNumber)
            {
                for (int i = 0; i < fingerNumber; i++)
                {
                    if(int.Parse(values[i + 1]) == 1)
                    {
                        touchedState[i] = true;
                    }
                    else
                    {
                        touchedState[i] = false;
                    }
                    
                }
            }
            else
            {
                Debug.LogFormat("reveice touch positions, data length: {0}", values.Length);
                Debug.LogFormat("touch state format is wrong: {0}", data);
            }
        }
    }
}

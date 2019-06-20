using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Es.InkPainter;

[RequireComponent(typeof(MRTouchServer2))]
public class Painter : MonoBehaviour {

    [Header("Painter")]
    public Brush brush;
    public Transform inkCanvas;
    public Transform inkCanvasCenter;

    [Header("MRTouch Coordinate")]
    public int mrWidth = 224;
    public int mrHeight = 171;

    [Header("Debug")]
    public Transform debugTouchedPoint;

    private InkCanvas canvas;
    private MRTouchServer2 server;

	// Use this for initialization
	void Start () {
        if (inkCanvas == null)
        {
            Destroy(this);
        }
        canvas = inkCanvas.GetComponent<InkCanvas>();
        server = GetComponent<MRTouchServer2>();
    }
	
	// Update is called once per frame
	void Update () {
        for(int i=0;i<server.fingerNumber;i++)
        {
            if (server.touchedState[i])
            {
                //Debug.LogFormat("touch index: {0}", i);
                Vector3 unityPos = MRTouchToUnityWorldPosition(server.fingertips[i], inkCanvasCenter.position, inkCanvas.localScale);
                if (debugTouchedPoint != null)
                {
                    debugTouchedPoint.position = unityPos;
                }
                canvas.Paint(brush, unityPos);
            }
        }
		
	}

    Vector3 MRTouchToUnityWorldPosition(Vector2 touchedPos, Vector3 unityCenter, Vector3 unityCanvasScale)
    {
        Vector2 normalizeTouchedPos = new Vector2(touchedPos.x / mrWidth, touchedPos.y / mrHeight);
        //Debug.LogFormat("touch pos: ({0}, {1})", touchedPos.x, touchedPos.y);
        //Debug.LogFormat("normalize touch pos: ({0}, {1})", normalizeTouchedPos.x, normalizeTouchedPos.y);
        Vector3 localPos = new Vector3(normalizeTouchedPos.x * unityCanvasScale.x, normalizeTouchedPos.y * -unityCanvasScale.y, 0);
        //Debug.LogFormat("loca pos: ({0}, {1}, {2})", localPos.x, localPos.y, localPos.z);
        return inkCanvasCenter.TransformPoint(localPos.x, localPos.y, localPos.z);
    }
}

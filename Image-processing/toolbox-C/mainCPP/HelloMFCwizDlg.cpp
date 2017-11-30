
// HelloMFCwizDlg.cpp : implementation file
//

#include "stdafx.h"
#include "HelloMFCwiz.h"
#include "HelloMFCwizDlg.h"
#include "afxdialogex.h"
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv2\videoio.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>

#include "opencv2\imgproc.hpp"
#include "opencv2\video/background_segm.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/stitching.hpp"
#include "opencv/util.hpp"

#include <cctype>
#include <fstream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

bool video_close;
bool morpho_close;
bool face_detection_close;
bool epipolar_close;
string filename;


//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

// Function Headers
void detectAndDisplay(Mat frame);
void printUsage();
int parseCmdArgs(int argc, char** argv);
bool try_use_gpu = false;
Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;
string result_name = "result";
// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "c:/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved

// CHelloMFCwizDlg dialog
// Returns an empty string if dialog is canceled
string openfilename(char *filter = "All Files (*.*)\0*.*\0", HWND owner = NULL) {
  OPENFILENAME ofn;
  char fileName[MAX_PATH] = "";
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(OPENFILENAME);
  ofn.hwndOwner = owner;
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = fileName;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
  ofn.lpstrDefExt = "";
  string fileNameStr;
  if ( GetOpenFileName(&ofn) )
    fileNameStr = fileName;
  return fileNameStr;
}

// help for morphology fucntion
static void help()
{
    printf("\n"
            "This program demonstrated a simple method of connected components clean up of background subtraction\n"
            "When the program starts, it begins learning the background.\n"
            "You can toggle background learning on and off by hitting the space bar.\n"
            "Call\n"
            "./segment_objects [video file, else it reads camera 0]\n\n");
}

static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat temp;
    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);
    findContours( temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    dst = Mat::zeros(img.size(), CV_8UC3);
    if( contours.size() == 0 )
        return;
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    Scalar color( 0, 0, 255 );
    drawContours( dst, contours, largestComp, color, FILLED, LINE_8, hierarchy );
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

// Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

// Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }

        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        // Form a filename
        filename = "";
        stringstream ssfn;
        ssfn << filenumber << ".png";
        filename = ssfn.str();
        filenumber++;

		stringstream ssroot;
		string folderName = "faces";
		//string folderCreateCommand = "mkdir " + folderName;

		//system(folderCreateCommand.c_str());

		ssroot<<folderName<<"/"<<filename;

		string fullPath = ssroot.str();
		ssroot.str("");

        imwrite(fullPath, gray);

        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }

// Show image
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    text = sstm.str();

    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    imshow("original", frame);

    if (!crop.empty())
    {
        imshow("detected", crop);
    }
    else
        destroyWindow("detected");
}

////////**** Video stiching subfunction********/////////
void printUsage()
{
    cout <<
        "Images stitcher.\n\n"
        "stitching img1 img2 [...imgN]\n\n"
        "Flags:\n"
        "  --try_use_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "  --mode (panorama|scans)\n"
        "      Determines configuration of stitcher. The default is 'panorama',\n"
        "      mode suitable for creating photo panoramas. Option 'scans' is suitable\n"
        "      for stitching materials under affine transformation, such as scans.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--try_use_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_use_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_use_gpu = true;
            else
            {
                cout << "Bad --try_use_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (string(argv[i + 1]) == "panorama")
                mode = Stitcher::PANORAMA;
            else if (string(argv[i + 1]) == "scans")
                mode = Stitcher::SCANS;
            else
            {
                cout << "Bad --mode flag value\n";
                return -1;
            }
            i++;
        }
        else
        {
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
        }
    }
    return 0;
}

////////////////////**** manin functionality of OPENCV****//////////////////////
// openCV operational and core fucntionality
CHelloMFCwizDlg::CHelloMFCwizDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CHelloMFCwizDlg::IDD, pParent)
	, my_edit1(_T("instruction"))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CHelloMFCwizDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, my_edit1);
}

BEGIN_MESSAGE_MAP(CHelloMFCwizDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CHelloMFCwizDlg::OnBnClickedOk)
	//ON_LBN_SELCHANGE(IDC_LIST1, &CHelloMFCwizDlg::OnLbnSelchangeList1)
	ON_BN_CLICKED(IDC_BUTTON1, &CHelloMFCwizDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CHelloMFCwizDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CHelloMFCwizDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CHelloMFCwizDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CHelloMFCwizDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CHelloMFCwizDlg::OnBnClickedButton6)
END_MESSAGE_MAP()


// CHelloMFCwizDlg message handlers

BOOL CHelloMFCwizDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CHelloMFCwizDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CHelloMFCwizDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CHelloMFCwizDlg::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
		//create matix to store image
	Mat image;
	VideoCapture cap;
	cap.open(0);
	/*if(!cap.isOpened())
	{
		my_edit1 = "the video has not been openned successfully";
		UpdateData(false);
	}
	else
	{
		my_edit1 = "the camera is launched successfully and video will be opened in few second. please wait";
		UpdateData(false);
	}*/

	namedWindow("window",1);
	while(video_close == false)
	{
		cap>>image;
		imshow("window",image);
		waitKey(33);
	}
	if (video_close ==  true)
	{
		cap.release();
		cvDestroyWindow("window");
	}
}


void CHelloMFCwizDlg::OnLbnSelchangeList1()
{
	// TODO: Add your control notification handler code here
}


void CHelloMFCwizDlg::OnBnClickedButton1()
{
	// TODO: Add your control notification handler code here
	cvDestroyAllWindows();
	video_close = true;
	face_detection_close =true;
	morpho_close =true;
	epipolar_close =true;
}


void CHelloMFCwizDlg::OnBnClickedButton2()
{
    VideoCapture capture(0);

    if (!capture.isOpened())  // check if we succeeded
        cout<<"the video has not been openned successfully";
		//GetDlgItem(IDC_EDIT1)->SetWindowTextA("the video has not been openned successfully");
    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
    }
	else
	{
		printf("Xml file containing faces was successfully loaded");
	};

    // Read the video stream
    Mat frame;

    while (face_detection_close == false)
    {
        capture >> frame;

        // Apply the classifier to the frame
        if (!frame.empty())
        {
            detectAndDisplay(frame);
        }
        else
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        int c = waitKey(10);

        if (27 == char(c))
        {
            break;
        }
    }
	if(face_detection_close == true)
	{
		capture.release();
		cvDestroyWindow("original");
		cvDestroyWindow("detected");
	}
}



void CHelloMFCwizDlg::OnBnClickedButton3()
{
	// TODO: Add your control notification handler code here
	string selectedImage;
	selectedImage=openfilename();

	/*my_edit1 = "you are going to load image from file";
	UpdateData(false);
	waitKey(2);
	my_edit1 = "please select an image it will be shown";
	UpdateData(false);*/
	//load image
	Mat image;
    image = imread( selectedImage, IMREAD_COLOR ); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
}


void CHelloMFCwizDlg::OnBnClickedButton4()
{
	// TODO: Add your control notification handler code here
	VideoCapture cap;
    bool update_bg_model = true;
	bool radiohelp=GetDlgItem(IDC_RADIO1)->IsDlgButtonChecked(IDC_RADIO1);
	//bool radiohelp = GetCheckedRadioButton(IDC_RADIO1);
    /*CommandLineParser parser("{help h||}{@input||}");
    if (parser.has("help"))
    {
        help();
    }
    string input = parser.get<std::string>("@input");
    if (input.empty())
        cap.open(0);
    else*/
    cap.open(0);
    if( !cap.isOpened() )
    {
        printf("\nCan not open camera or video file\n");
    }
    Mat tmp_frame, bgmask, out_frame;
    cap >> tmp_frame;
    if(tmp_frame.empty())
    {
        printf("can not read data from the video source\n");
    }
    namedWindow("video", 1);
    namedWindow("segmented", 1);
    Ptr<BackgroundSubtractorMOG2> bgsubtractor=createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(10);
    while(morpho_close==false)
    {
        cap >> tmp_frame;
        if( tmp_frame.empty() )
            break;
        bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
        refineSegments(tmp_frame, bgmask, out_frame);
        imshow("video", tmp_frame);
        imshow("segmented", out_frame);
        char keycode = (char)waitKey(30);
        if( keycode == 27 )
            break;
        if( keycode == ' ' )
        {
            update_bg_model = !update_bg_model;
            printf("Learn background is in state = %d\n",update_bg_model);
        }
    }
	if(morpho_close==true)
	{
		cap.release();
		cvDestroyWindow("video");
		cvDestroyWindow("segmented");
	}
}


void CHelloMFCwizDlg::OnBnClickedButton5()
{
	// TODO: Add your control notification handler code here
	//int retval = parseCmdArgs(argc, argv);
	//GetDlgItem(IDC_EDIT1)->SetWindowTextA("instruction to run morphological operation");
	//load first image
	Mat image1;
	string selectedImage;
	selectedImage=openfilename();
    image1 = imread( selectedImage, IMREAD_COLOR ); // Read the file
    if( image1.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    namedWindow( "Display window 1", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window 1", image1 );                // Show our image inside it.
    waitKey(10); // Wait for a keystroke in the window
	imgs.push_back(image1);
	Mat image2;
	selectedImage=openfilename();
    image2 = imread( selectedImage, IMREAD_COLOR ); // Read the file
    if( image2.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    namedWindow( "Display window 2", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window 2", image2 );                // Show our image inside it.
    waitKey(10); // Wait for a keystroke in the window
	imgs.push_back(image2);

	// start stiching process
    Mat pano;
	mode = Stitcher::PANORAMA;
    Ptr<Stitcher> stitcher = Stitcher::create(mode, try_use_gpu);
    Stitcher::Status status = stitcher->stitch(imgs, pano);

    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
    }
	else
	{
		namedWindow( "Display window result", WINDOW_AUTOSIZE ); // Create a window for display.
		imshow( "Display window result", pano );
		//waitKey(5);
		string filename_res;
		filename_res = "";
		stringstream ssfn;
		ssfn << result_name << ".png";
		filename = ssfn.str();
		cout<<"file name is:"<<filename;
		imwrite("result.png", pano);
	}
}


void CHelloMFCwizDlg::OnBnClickedButton6()
{
	// TODO: Add your control notification handler code here
	    /****************** EPIPOLAR GEOMETRY **************************/

   // Mat img_1 = imread("img/img1.png");
   // Mat img_2 = imread("img/img2.png");

    //Mat img_1 = imread("img/perra_7.jpg");
    //Mat img_2 = imread("img/perra_8.jpg");

    // Mat img_1 = imread("img/madera_1.jpg");
    // Mat img_2 = imread("img/madera_2.jpg");

	/*imshow("image1", img_1);
	waitKey(10);
	imshow("image2", img_2);
	waitKey(10);*/

	Mat img_1, img_2;
	string selectedImage;
	selectedImage=openfilename();
    img_1 = imread( selectedImage, IMREAD_COLOR ); // Read the file
    if( img_1.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
	selectedImage=openfilename();
    img_2 = imread( selectedImage, IMREAD_COLOR ); // Read the file
    if( img_2.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
    }
    Mat fund_mat = Mat::zeros(3,3,CV_64F);

    Vec3d epipole;

    vector<Vec3d> lines_1, lines_2;
    vector<Point2d> good_matches_1, good_matches_2;

    // Get epipolar geometry
    computeEpiLines(img_1, img_2,
                    epipole, fund_mat,
                    lines_1, lines_2,
                    good_matches_1, good_matches_2);

    Mat A, B, Ap, Bp;

    Mat e_x = crossProductMatrix(epipole);

    /****************** PROJECTIVE **************************/

    // Get A,B matrix for minimizing z
    obtainAB(img_1, e_x, A, B);
    obtainAB(img_2, fund_mat, Ap, Bp);

    // Get initial guess for z
    Vec3d z = getInitialGuess(A, B, Ap, Bp);

    // Optimizes the z solution
    optimizeRoot(A, B, Ap, Bp, z);

    // Get w
    Mat w = e_x * Mat(z);
    Mat wp = fund_mat * Mat(z);

    w /= w.at<double>(2,0);
    wp /= wp.at<double>(2,0);

    // Get final H_p and Hp_p matrix for projection
    Mat H_p = Mat::eye(3, 3, CV_64F);
    H_p.at<double>(2,0) = w.at<double>(0,0);
    H_p.at<double>(2,1) = w.at<double>(1,0);

    Mat Hp_p = Mat::eye(3, 3, CV_64F);
    Hp_p.at<double>(2,0) = wp.at<double>(0,0);
    Hp_p.at<double>(2,1) = wp.at<double>(1,0);

    /****************** SIMILARITY **************************/

    // Get the translation term
    double vp_c = getTranslationTerm(img_1, img_2, H_p, Hp_p);

    // Get the H_r and Hp_r matrix directly
    Mat H_r = Mat::zeros(3, 3, CV_64F);

    H_r.at<double>(0,0) = fund_mat.at<double>(2,1) - w.at<double>(1,0) * fund_mat.at<double>(2,2);
    H_r.at<double>(1,0) = fund_mat.at<double>(2,0) - w.at<double>(0,0) * fund_mat.at<double>(2,2);

    H_r.at<double>(0,1) = w.at<double>(0,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(2,0);
    H_r.at<double>(1,1) = H_r.at<double>(0,0);

    H_r.at<double>(1,2) = fund_mat.at<double>(2,2) + vp_c;
    H_r.at<double>(2,2) = 1.0;

    Mat Hp_r = Mat::zeros(3, 3, CV_64F);

    Hp_r.at<double>(0,0) = wp.at<double>(1,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(1,2);
    Hp_r.at<double>(1,0) = wp.at<double>(0,0) * fund_mat.at<double>(2,2) - fund_mat.at<double>(0,2);

    Hp_r.at<double>(0,1) = fund_mat.at<double>(0,2) - wp.at<double>(0,0) * fund_mat.at<double>(2,2);
    Hp_r.at<double>(1,1) = Hp_r.at<double>(0,0);

    Hp_r.at<double>(1,2) = vp_c;
    Hp_r.at<double>(2,2) = 1.0;

    /******************* SHEARING ***************************/

    Mat H_1 = H_r*H_p;
    Mat H_2 = Hp_r*Hp_p;

    Mat H_s, Hp_s;

    // Get shearing transforms with the method described on the paper
    getShearingTransforms(img_1, img_2, H_1, H_2, H_s, Hp_s);

    /****************** RECTIFY IMAGES **********************/

    Mat H = H_s * H_r * H_p;
    Mat Hp = Hp_s * Hp_r * Hp_p;


            // Get homography image of the corner coordinates from all the images
            vector<Point2d> corners_all(4), corners_all_t(4);
            double min_x, min_y, max_x, max_y;
            min_x = min_y = +INF;
            max_x = max_y = -INF;

            corners_all[0] = Point2d(0,0);
            corners_all[1] = Point2d(img_1.cols,0);
            corners_all[2] = Point2d(img_1.cols,img_1.rows);
            corners_all[3] = Point2d(0,img_1.rows);

            perspectiveTransform(corners_all, corners_all_t, H);

            for (int j = 0; j < 4; j++) {
                min_x = min(corners_all_t[j].x, min_x);
                max_x = max(corners_all_t[j].x, max_x);

                min_y = min(corners_all_t[j].y, min_y);
                max_y = max(corners_all_t[j].y, max_y);
            }

            int img_1_cols = max_x - min_x;
            int img_1_rows = max_y - min_y;

            // Get homography image of the corner coordinates from all the images
            min_x = min_y = +INF;
            max_x = max_y = -INF;

            corners_all[0] = Point2d(0,0);
            corners_all[1] = Point2d(img_2.cols,0);
            corners_all[2] = Point2d(img_2.cols,img_2.rows);
            corners_all[3] = Point2d(0,img_2.rows);

            perspectiveTransform(corners_all, corners_all_t, Hp);

            for (int j = 0; j < 4; j++) {
                min_x = min(corners_all_t[j].x, min_x);
                max_x = max(corners_all_t[j].x, max_x);

                min_y = min(corners_all_t[j].y, min_y);
                max_y = max(corners_all_t[j].y, max_y);
            }

            int img_2_cols = max_x - min_x;
            int img_2_rows = max_y - min_y;

    // Apply homographies
    Mat img_1_dst(img_1_rows, img_1_cols, CV_64F);
    Mat img_2_dst(img_2_rows, img_2_cols, CV_64F);

    warpPerspective( img_1, img_1_dst, H, img_1_dst.size() );
    warpPerspective( img_2, img_2_dst, Hp, img_2_dst.size() );

    Vec3d epipole_dst;

    vector<Vec3d> lines_1_dst, lines_2_dst;
    vector<Point2d> good_matches_1_dst, good_matches_2_dst;

    perspectiveTransform(good_matches_1, good_matches_1_dst, H);
    perspectiveTransform(good_matches_2, good_matches_2_dst, Hp);

    // Get epipolar geometry and draw epilines
    computeEpiLines(img_1_dst, img_2_dst, epipole_dst, fund_mat, lines_1_dst, lines_2_dst, good_matches_1_dst, good_matches_2_dst);

    drawEpilines(img_1, img_2, lines_1, lines_2, good_matches_1, good_matches_2, 150);
    drawEpilines(img_1_dst, img_2_dst, lines_1_dst, lines_2_dst, good_matches_1_dst, good_matches_2_dst, 150);

    cout << "\nH = " << H << "\nHp = " << Hp << endl;

    cout << "\nEpipolo antes: " << epipole/epipole[2] << "\nEpipolo después: " << epipole_dst << endl;

    draw(img_1, "1");
    draw(img_1_dst, "1 rectificada");

    char c = 'a';

    draw(img_2, "2");
    draw(img_2_dst, "2 rectificada");

    c = 'a';

    while (epipolar_close= false)
      c = waitKey();
	if(epipolar_close == true)
	{
    destroyAllWindows();
	}
}

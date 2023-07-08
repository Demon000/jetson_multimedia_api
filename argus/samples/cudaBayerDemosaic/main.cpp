/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Argus/Argus.h>

#include "ArgusHelpers.h"
#include "CommonOptions.h"
#include "CudaBayerDemosaicConsumer.h"

#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

namespace ArgusSamples
{

// Globals and derived constants.
EGLDisplayHolder g_display;

bool shouldExit = false;

void sigIntHandlerFn(int s) {
    shouldExit = true;
}

class SampleOptions : public CommonOptions
{
public:
    SampleOptions(const char *programName)
        : CommonOptions(programName,
                        ArgusSamples::CommonOptions::Option_M_SensorMode |
                        ArgusSamples::CommonOptions::Option_R_WindowRect |
                        ArgusSamples::CommonOptions::Option_T_CaptureTime)
        , m_numStreams(1)
    {
        m_captureTime.set(0);

        addOption(createValueOption
            ("num", 'n', "COUNT", "Number of streams", m_numStreams));
    }

    uint32_t numStreams() const { return m_numStreams.get(); }

protected:
    Value<uint32_t> m_numStreams;
};


class CaptureHolder
{
public:
    explicit CaptureHolder() {}
    virtual ~CaptureHolder() {}

    bool initializeBeforePreview(
        const SampleOptions& options,
        ICameraProvider *iCameraProvider,
        CameraDevice *cameraDevice);
    bool initializeAfterPreview();
    bool shutdownBeforePreview();
    bool shutdownBeforePreview2();
    bool shutdownBeforePreview3();
    bool shutdownAfterPreview();

    EGLStreamKHR getOutputStream() const {
        return cudaConsumer->getOutputStream();
    }

private:
    CudaBayerDemosaicConsumer *cudaConsumer;
    CaptureSession *captureSession;
    ICaptureSession *iCaptureSession;
    OutputStreamSettings *streamSettings;
    OutputStream *outputStream;
    Request *request;
};


/**
 * Main thread function opens connection to Argus driver, creates a capture session for
 * a given camera device and sensor mode, then creates a RAW16 stream attached to a
 * CudaBayerConsumer such that the CUDA consumer will acquire the outputs of capture
 * results as raw Bayer data (which it then demosaics to RGBA for demonstration purposes).
 */
static bool execute(const SampleOptions& options)
{
    // Initialize the preview window and EGL display.
    Window &window = Window::getInstance();
    window.setWindowRect(options.windowRect());
    PROPAGATE_ERROR(g_display.initialize(window.getEGLNativeDisplay()));

    // Create the Argus CameraProvider object
    UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
    {
        ORIGINATE_ERROR("Failed to create CameraProvider");
    }
    printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

    /* Get the camera devices */
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);

    if (cameraDevices.size() == 0)
        ORIGINATE_ERROR("No cameras available");

    printf("Camera devices: %lu\n", cameraDevices.size());

    uint32_t streamCount = cameraDevices.size();
    if (streamCount > options.numStreams())
        streamCount = options.numStreams();

    printf("Streams: %u\n", streamCount);

    std::vector<CaptureHolder> captureHolders;
    std::vector<EGLStreamKHR> rgbaOutputStreams;
    for (auto& cameraDevice : cameraDevices) {
        if (captureHolders.size() >= streamCount)
            break;

        captureHolders.emplace_back();
        auto& captureHolder = captureHolders.back();
        PROPAGATE_ERROR(captureHolder.initializeBeforePreview(
            options, iCameraProvider, cameraDevice));
        rgbaOutputStreams.push_back(captureHolder.getOutputStream());
    }

    PreviewConsumerThread previewConsumer(g_display.get(),
                                          rgbaOutputStreams,
                                          PreviewConsumerThread::RenderLayout::LAYOUT_TILED);
    PROPAGATE_ERROR(previewConsumer.initialize());
    PROPAGATE_ERROR(previewConsumer.waitRunning());

   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = sigIntHandlerFn;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;

   sigaction(SIGINT, &sigIntHandler, NULL);

    for (auto& captureHolder : captureHolders) {
        captureHolder.initializeAfterPreview();
    }

    if (options.captureTime()) {
        PROPAGATE_ERROR(window.pollingSleep(options.captureTime()));
    } else {
        while (!shouldExit) {
            PROPAGATE_ERROR(window.pollingSleep(1));
        }
    }

    for (auto& captureHolder : captureHolders) {
        captureHolder.shutdownBeforePreview();
    }

    for (auto& captureHolder : captureHolders) {
        captureHolder.shutdownBeforePreview2();
    }

    for (auto& captureHolder : captureHolders) {
        captureHolder.shutdownBeforePreview3();
    }

    PROPAGATE_ERROR(previewConsumer.shutdown());

    for (auto& captureHolder : captureHolders) {
        captureHolder.shutdownAfterPreview();
    }

    // Shut down Argus.
    cameraProvider.reset();

    // Shut down the window (destroys window's EGLSurface).
    window.shutdown();

    // Cleanup the EGL display
    PROPAGATE_ERROR(g_display.cleanup());

    return true;
}

bool CaptureHolder::initializeBeforePreview(
    const SampleOptions& options,
    ICameraProvider *iCameraProvider,
    CameraDevice *cameraDevice)
{
    SensorMode* sensorMode = ArgusHelpers::getSensorMode(cameraDevice, options.sensorModeIndex());
    ISensorMode *iSensorMode = interface_cast<ISensorMode>(sensorMode);
    if (!iSensorMode)
        ORIGINATE_ERROR("Selected sensor mode not available");

    // Create the capture session using the selected device.
    captureSession = (iCameraProvider->createCaptureSession(cameraDevice));
    iCaptureSession = interface_cast<ICaptureSession>(captureSession);
    if (!iCaptureSession)
    {
        ORIGINATE_ERROR("Failed to create CaptureSession");
    }

    // Create the RAW16 output EGLStream using the sensor mode resolution.
    streamSettings = (
        iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
    IEGLOutputStreamSettings *iEGLStreamSettings =
        interface_cast<IEGLOutputStreamSettings>(streamSettings);
    if (!iEGLStreamSettings)
    {
        ORIGINATE_ERROR("Failed to create OutputStreamSettings");
    }
    iEGLStreamSettings->setEGLDisplay(g_display.get());
    iEGLStreamSettings->setPixelFormat(PIXEL_FMT_RAW16);
    iEGLStreamSettings->setResolution(iSensorMode->getResolution());
    iEGLStreamSettings->setMode(EGL_STREAM_MODE_FIFO);
    outputStream = (iCaptureSession->createOutputStream(streamSettings));
    IEGLOutputStream *iEGLOutputStream = interface_cast<IEGLOutputStream>(outputStream);
    if (!iEGLOutputStream)
    {
        ORIGINATE_ERROR("Failed to create EGLOutputStream");
    }

    // Create capture request and enable output stream.
    request = (iCaptureSession->createRequest());
    IRequest *iRequest = interface_cast<IRequest>(request);
    if (!iRequest)
    {
        ORIGINATE_ERROR("Failed to create Request");
    }
    iRequest->enableOutputStream(outputStream);

    // Set the sensor mode in the request.
    ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(request);
    if (!iSourceSettings)
        ORIGINATE_ERROR("Failed to get source settings request interface");
    iSourceSettings->setSensorMode(sensorMode);

    // Create the CUDA Bayer consumer and connect it to the RAW16 output stream.
    cudaConsumer = new CudaBayerDemosaicConsumer(iEGLOutputStream->getEGLDisplay(),
                                                 iEGLOutputStream->getEGLStream(),
                                                 iEGLStreamSettings->getResolution());
    PROPAGATE_ERROR(cudaConsumer->initialize());
    PROPAGATE_ERROR(cudaConsumer->waitRunning());

    PROPAGATE_ERROR(cudaConsumer->initializeBeforePreview());

    return true;
}

bool CaptureHolder::initializeAfterPreview()
{
    PROPAGATE_ERROR(cudaConsumer->initializeAfterPreview());

    if (iCaptureSession->repeat(request) != STATUS_OK) {
        ORIGINATE_ERROR("Failed to start repeat capture request");
    }

    return true;
}

bool CaptureHolder::shutdownBeforePreview()
{
    iCaptureSession->stopRepeat();

    return true;
}

bool CaptureHolder::shutdownBeforePreview2()
{
    // Wait until all captures have completed.
    iCaptureSession->waitForIdle();

    cudaConsumer->stopCapture();

    return true;
}

bool CaptureHolder::shutdownBeforePreview3()
{
    // Shutdown the CUDA consumer.
    PROPAGATE_ERROR(cudaConsumer->shutdownBeforePreview());

    return true;
}

bool CaptureHolder::shutdownAfterPreview()
{
    outputStream->destroy();
    streamSettings->destroy();
    request->destroy();
    captureSession->destroy();

    PROPAGATE_ERROR(cudaConsumer->shutdownAfterPreview());
    PROPAGATE_ERROR(cudaConsumer->shutdown());

    /* Destroy the output stream */
    delete cudaConsumer;

    return true;
}

}; // namespace ArgusSamples

int main(int argc, char** argv)
{
    printf("Executing Argus Sample: %s\n", basename(argv[0]));

    ArgusSamples::SampleOptions options(basename(argv[0]));

    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    if (!ArgusSamples::execute(options))
        return EXIT_FAILURE;

    printf("Argus sample '%s' completed successfully.\n", basename(argv[0]));

    return EXIT_SUCCESS;
}

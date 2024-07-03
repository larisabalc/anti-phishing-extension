import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';
import browser from 'webextension-polyfill';

function getTextColorClass(probability) {
  if (probability > 0.75) {
    return 'high-probability';
  } else if (probability > 0.5) {
    return 'moderate-probability';
  } else {
    return 'low-probability';
  }
}

function renderProbabilityIcon(probability) {
  if (probability > 0.75) {
    return <i className="fas fa-times"></i>;
  } else if (probability > 0.5 && probability < 0.75){
    return <i className="fas fa-exclamation-triangle"></i>;
  }
  else return <i className="fas fa-check"></i>;
}

function getTextColorClassWithoutProbability(result) {
  if (result === 'phishing' || result === 'malicious') {
    return 'high-probability';
  } else {
    return 'low-probability';
  }
}

function renderProbabilityIconWithoutProbability(result) {
  if (result === 'phishing') {
    return <i className="fas fa-times"></i>;
  } else {
    return <i className="fas fa-check"></i>;
  }
}

function getCurrentTab(callback) {
  browser.tabs.query({ active: true, currentWindow: true })
    .then(tabs => {
      if (tabs && tabs.length > 0) {
        callback(tabs[0]);
        return;
      } else {
        console.error('Unable to get current tab');
        callback(null);
      }
    })
    .catch(error => {
      console.error('Error fetching current tab information:', error);
      callback(null);
    });
}

const CurrentTabInfo = () => {
  const [url, setUrl] = useState({});
  const [urlLoading, setUrlLoading] = useState(false);
  const [ipInfo, setIpInfo] = useState(null);
  const [ipLoading, setIpLoading] = useState(false);
  const [currentTab, setCurrentTab] = useState(null);
  const [waitingForRetrain, setWaitingForRetrain] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  useEffect(() => {
    const handleMessage = (message) => {
      try {
        const responseObject = message;

        if (responseObject.type === 'TrainingProgress') {
          setTrainingProgress(responseObject.progress);
        } else if (responseObject.type === 'RetrainFinished') {
          setWaitingForRetrain(false);
        } else if (responseObject.type === 'CurrentTab' || responseObject.type === 'PreCheck') {
          console.log('Message received from background script:', message);

          const { target_proba, result, domain, ip_address, country, organization, product } = responseObject;

          setUrl({ target_proba, result });
          setUrlLoading(false);
          setIpInfo({ ip_address, domain, country, organization, product });
          setIpLoading(false);
        }
      } catch (error) {
        console.error('Error parsing message from background script:', error);
      }
    };

    browser.runtime.onMessage.addListener(handleMessage);

    return () => {
      browser.runtime.onMessage.removeListener(handleMessage);
    };
  }, []);

  useEffect(() => {
    fetchCurrentTabInfo();
  }, []);

  const sendURLToBackground = useCallback((urls, messageType) => {
    try {
      browser.runtime.sendMessage({ urls: urls, type: messageType }).then((response) => {
        console.log('URLs sent to background script');
      });
    } catch (error) {
      console.error('Error sending message to background script:', error);
    }
  }, []);

  const sendFalseNegative = useCallback((url, result, type) => {
    try {
      browser.runtime.sendMessage({ url, result, type }).then((response) => {
        console.log('URL results sent to background script');
      });
    } catch (error) {
      console.error('Error sending false negative message to background script:', error);
    }
  }, []);

  const handleReportFalseNegative = useCallback(() => {
    setWaitingForRetrain(true);
    sendFalseNegative(currentTab, url.result, 'FalseNegative');
  }, [currentTab, url.result, sendFalseNegative]);

  const fetchCurrentTabInfo = useCallback(() => {
    getCurrentTab(tab => {
      if (tab) {
        const url = new URL(tab.url);
        console.log(url);

        let tabUrl;
        if (url.searchParams.has('referrer')) {
          const referrer = decodeURIComponent(url.searchParams.get('referrer'));
          tabUrl = referrer;
        } else {
          tabUrl = tab.url;
        }

        setCurrentTab(tabUrl);
        setUrlLoading(true);
        setIpLoading(true);
        sendURLToBackground(tabUrl, url.searchParams.has('referrer') ? 'PreCheck' : 'CurrentTab');
      } else {
        console.error('Error fetching current tab information: Unable to get current tab');
      }
    });
  }, [currentTab, setCurrentTab, setUrlLoading, setIpLoading, sendURLToBackground]);

  return (
    <div>
      <div className='row'>
          <div className='col-md-12'>
            <div className='url-result'>
              <div className="col-md-10">
                <p className="section-title-white align-items-center">URL: 
                  <div style={{ maxWidth:'300px', fontSize:'0.8rem', overflowX: 'auto' }}>
                    <a href={currentTab} target="_blank" rel="noopener noreferrer">{currentTab}</a>
                  </div>
                </p>
              </div>
            </div>
          </div>
      </div>
      {urlLoading ? (
        <div className='row'>
          <div className='col-md-12'>
            <div className='url-result loading'>
              <div className="col-md-10">
                <p className="section-title-white align-items-center">URL Risk:</p>
                <p className='white-text text-center'><i className="fas fa-spinner fa-spin"></i> Loading URL risk information...</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        url && (
          <div className={`url-result ${getTextColorClass(url.target_proba)}`}>
            <div className="row">
              <div className="col-md-12">
                <div className="d-flex align-items-center">
                  <div className="col-md-6">
                    <p className="section-title-white">URL Risk:</p>
                  </div>
                  <div className="col-md-6 text-center">
                    {waitingForRetrain ? (
                      <>
                        {trainingProgress > 0 && (
                          <div className="progress mt-3">
                            <div
                              className="progress-bar"
                              role="progressbar"
                              style={{ width: `${trainingProgress}%`, backgroundColor: '#00BFFF' }}
                              aria-valuenow={trainingProgress}
                              aria-valuemin="0"
                              aria-valuemax="100"
                            >
                              {parseInt(trainingProgress)}%
                            </div>
                          </div>
                        )}
                        <p className='white-text text-center'><i className="fas fa-spinner fa-spin"></i> Waiting...</p>
                      </>
                    ) : (
                      <button className="btn btn-danger mb-2" onClick={handleReportFalseNegative}>Report False</button>
                    )}
                  </div>
                </div>
              </div>
            </div>
            <div className="row">
              <div className="col-md-12">
                <div className="d-flex align-items-center">
                  <div className={`col-md-6 url-analysis ${getTextColorClass(url.target_proba)}`}>
                    <p>Probability: {url.target_proba}</p>
                    <p>Result: {url.result}</p>
                  </div>
                  <div className="col-md-6 text-center">
                    {renderProbabilityIcon(url.target_proba)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      )}
      {ipLoading ? (
        <div className='row'>
          <div className='col-md-12'>
            <div className='url-result loading'>
              <div className="col-md-10">
                <p className="section-title-white align-items-center">IP Details:</p>
                <p className='white-text text-center'><i className="fas fa-spinner fa-spin"></i> Loading IP information...</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        ipInfo && (
          <div className='row'>
            <div className='col-md-12'>
              <div className='url-result'>
                <div className="col-md-10">
                  <p className="section-title-white align-items-center">IP Details:</p>
                </div>
                <p className='white-text'><i class="fas fa-map-marker-alt"></i> IP Address: {ipInfo.ip_address}</p>
                <p className='white-text'><i class="fas fa-globe"></i> Domain: {ipInfo.domain}</p>
                <p className='white-text'><i class="fas fa-globe-americas"></i> Country: {ipInfo.country}</p>
                <p className='white-text'><i class="fas fa-building"></i> Organization: {ipInfo.organization}</p>
                <p className='white-text'><i class="fas fa-cogs"></i> Product: {ipInfo.product}</p>
              </div>
            </div>
          </div>
        )
      )}
    </div>
  );
};

const FetchUrls = () => {
  const [urlsInfo, setUrlsInfo] = useState({});
  const [loading, setLoading] = useState(false);
  const [showUrls, setShowUrls] = useState(false);
  const [tabHasReferrer, setTabHasReferrer] = useState(false);

  const handleMessage = (messages) => {
    if (messages.type === 'AnalyzedFetchedUrls') {
      console.log('Message received from background script:', messages);
      const updatedUrlsInfo = { ...urlsInfo };
      for (const index in messages.urls) {
        const { url, result } = messages.urls[index];
        updatedUrlsInfo[url] = result;
      }
      setUrlsInfo(updatedUrlsInfo);
      setLoading(false);
    }
  };

  useEffect(() => {
    browser.runtime.onMessage.addListener(handleMessage);
    return () => {
      browser.runtime.onMessage.removeListener(handleMessage);
    };
  }, [urlsInfo]);

  const sendURLsToBackground = useCallback((urls) => {
    browser.runtime.sendMessage({ urls, type: 'FetchUrls' }).then((response) => {
      console.log('URLs sent to background script for analysis:', urls);
    });
  }, []);

  const isResourceURL = (url) => {
    const resourceExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.ico'];
    return resourceExtensions.some(ext => url.includes(ext));
  };

  const handleImproveSecurityClick = async () => {
    setLoading(true);
    try {
      if (!showUrls) {
        getCurrentTab(async (tab) => {
          if (tab) {
            const response = await fetch(tab.url);
            const html = await response.text();
            const regex = /href="(https?:\/\/[^"]+)"/g;
            const foundUrls = [];
            let match;
            while ((match = regex.exec(html)) !== null) {
              if (!isResourceURL(match[1])) {
                foundUrls.push(match[1]);
              }
            }
            if (foundUrls.length > 0) {
              sendURLsToBackground(foundUrls);
            } else {
              setLoading(false);
              setUrlsInfo({ noUrls: 'No URLs found' });
            }
          } else {
            console.error('Error fetching current tab information: Unable to get current tab');
          }
        });
      }
      setShowUrls(!showUrls);
    } catch (error) {
      console.error('Error fetching URLs:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    const checkTabReferrer = () => {
      getCurrentTab((tab) => {
        if (tab) {
          const url = new URL(tab.url);
          if (url.searchParams.has('referrer')) {
            setTabHasReferrer(true);
          }
        } else {
          console.error('Error fetching current tab information: Unable to get current tab');
        }
      });
    };
    checkTabReferrer();
  }, []);

  return (
    <div>
      {!tabHasReferrer && (
        <div>
          <button className="btn btn-primary btn-block" onClick={handleImproveSecurityClick}>
            {showUrls ? 'Hide' : 'Improve Security'}
          </button>
          {showUrls && (
            <div className={`url-result ${loading ? 'loading' : ''}`}>
              <div className="col-md-12">
                <p className="section-title-white align-items-center">URLs fetched from current tab:</p>
              </div>
              {loading ? (
                <p className='white-text text-center'> <i className="fas fa-spinner fa-spin"></i> Fetching and analyzing URLs...</p>
              ) : (
                Object.keys(urlsInfo).length === 0 || urlsInfo.noUrls ? (
                  <p className='white-text text-center'>No URLs Fetched from curent tab.</p>
                ) : (
                  <div className="table-responsive">
                    <table className="table">
                      <thead className='align-items-center text-center'>
                        <tr>
                          <th className='white-text'>URL</th>
                          <th className='white-text'>Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(urlsInfo).map(([url, result], index) => (
                          <tr key={index} className={getTextColorClassWithoutProbability(result)}>
                            <td style={{ maxWidth: '200px' }}>
                              <div style={{ overflowX: 'auto' }}>
                                <a href={url} target="_blank" rel="noopener noreferrer">
                                  {url}
                                </a>
                              </div>
                            </td>
                            <td className='text-center'>{result}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const MessageSender = () => {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleMessage = (message) => {
    const responseObject = message;

    if (responseObject.type === 'CheckedEmail') {
      console.log('Message received from background script:', message);
      setResult(responseObject.result);
      setLoading(false);
    }
  };

  useEffect(() => {
    browser.runtime.onMessage.addListener(handleMessage);

    return () => {
      browser.runtime.onMessage.removeListener(handleMessage);
    };
  }, []);

  const sendEmailToBackground = useCallback((email) => {
    setLoading(true);
    browser.runtime.sendMessage({ email: email, type: 'CheckEmail' }).then((response) => {
      console.log('Email sent to background script for analysis:', email);
    });
  },[]);

  const handleMessageChange = (event) => {
    setMessage(event.target.value);
  };

  const handleSendClick = async () => {
    setLoading(false);
    if (message.trim() !== '') {
      sendEmailToBackground(message);
    }
  };

  return (
    <div>
      <textarea
        value={message}
        onChange={handleMessageChange}
        placeholder="Enter the email for analysis..."
        disabled={loading}
      />
      <button className='btn btn-primary btn-block' onClick={handleSendClick} disabled={loading}>
        {loading ? 'Sending...' : 'Send'}
      </button>
      {result && (
        <div className={`url-result ${loading ? 'loading' : ''}`}>
          <div className="row">
            <div className="col-md-12">
              {loading ? (
                <p className='white-text text-center'><i className="fas fa-spinner fa-spin"></i> Loading...</p>
              ):(
                <div className="url-info d-flex align-items-center">
                <div className={`col-md-10 text-center ${getTextColorClassWithoutProbability(result)}`}>
                  <p>This email is: {result}!</p>
                </div>
                <div className={`col-md-2 align-items-center text-center ${getTextColorClassWithoutProbability(result)}`} >
                  {renderProbabilityIconWithoutProbability(result)}
                </div>
              </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

ReactDOM.render(
  <div>
    <CurrentTabInfo/>
    <FetchUrls/>
  </div>,
  document.getElementById('react-target-url')
);

ReactDOM.render(
  <div>
    <MessageSender />
  </div>,
  document.getElementById('react-target-email')
);
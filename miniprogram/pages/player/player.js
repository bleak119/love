// pages/player/player.js
const app = getApp();

Page({
  data: {
    music: {},
    musicId: null,
    isPlaying: false,
    currentTime: '0:00',
    totalTime: '0:00',
    currentPosition: 0,
    isFavorite: false,
    showPlaylist: false,
    currentPlaylist: [],
    currentMusicIndex: 0,
    audioContext: null
  },

  onLoad(options) {
    const musicId = options.id;
    this.setData({ musicId });
    this.loadMusic(musicId);
    this.checkFavorite(musicId);
  },

  onUnload() {
    if (this.data.audioContext) {
      this.data.audioContext.stop();
    }
    wx.getBackgroundAudioManager().stop();
  },

  loadMusic(musicId) {
    this.setData({ 'music.id': musicId });

    // Get music details
    app.request({
      url: `/music/${musicId}`,
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          this.setData({ music: res.data });
          const minutes = Math.floor(res.data.durationSec / 60);
          const seconds = res.data.durationSec % 60;
          this.setData({
            totalTime: `${minutes}:${seconds.toString().padStart(2, '0')}`
          });
        }
      }
    });
  },

  checkFavorite(musicId) {
    app.request({
      url: `/favorites/${musicId}/check`,
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          this.setData({ isFavorite: res.data });
        }
      }
    });
  },

  togglePlay() {
    if (!this.data.isPlaying) {
      this.startPlaying();
    } else {
      this.pausePlaying();
    }
  },

  startPlaying() {
    app.request({
      url: `/music/${this.data.musicId}/play-url`,
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          const bgAudioManager = wx.getBackgroundAudioManager();
          bgAudioManager.title = this.data.music.title;
          bgAudioManager.singer = this.data.music.artist;
          bgAudioManager.coverImgUrl = this.data.music.coverUrl || '';
          bgAudioManager.src = res.data.playUrl;

          this.setData({ isPlaying: true });

          // Update progress every second
          this.progressTimer = setInterval(() => {
            this.updateProgress();
          }, 1000);
        }
      }
    });
  },

  pausePlaying() {
    wx.getBackgroundAudioManager().pause();
    this.setData({ isPlaying: false });
    clearInterval(this.progressTimer);
  },

  updateProgress() {
    const bgAudioManager = wx.getBackgroundAudioManager();
    this.setData({
      currentPosition: Math.floor(bgAudioManager.currentTime) || 0
    });

    const minutes = Math.floor(bgAudioManager.currentTime / 60) || 0;
    const seconds = Math.floor(bgAudioManager.currentTime % 60) || 0;
    this.setData({
      currentTime: `${minutes}:${seconds.toString().padStart(2, '0')}`
    });

    // Report progress to server every 30 seconds
    if (Math.floor(bgAudioManager.currentTime) % 30 === 0) {
      this.reportProgress();
    }
  },

  reportProgress() {
    const bgAudioManager = wx.getBackgroundAudioManager();
    app.request({
      url: `/music/${this.data.musicId}/report-progress?progressSec=${Math.floor(bgAudioManager.currentTime)}`,
      method: 'POST'
    });
  },

  handleSliderChange(e) {
    const position = e.detail.value;
    const bgAudioManager = wx.getBackgroundAudioManager();
    bgAudioManager.seek(position);
    this.setData({ currentPosition: position });
  },

  toggleFavorite() {
    const method = this.data.isFavorite ? 'DELETE' : 'POST';
    app.request({
      url: `/favorites/${this.data.musicId}`,
      method,
      success: (res) => {
        if (res.code === 0) {
          this.setData({ isFavorite: !this.data.isFavorite });
          wx.showToast({
            title: this.data.isFavorite ? 'Added to favorites' : 'Removed from favorites',
            icon: 'success',
            duration: 1500
          });
        }
      }
    });
  },

  previousMusic() {
    if (this.data.currentMusicIndex > 0) {
      const newIndex = this.data.currentMusicIndex - 1;
      this.playFromPlaylist({ currentTarget: { dataset: { index: newIndex } } });
    }
  },

  nextMusic() {
    if (this.data.currentMusicIndex < this.data.currentPlaylist.length - 1) {
      const newIndex = this.data.currentMusicIndex + 1;
      this.playFromPlaylist({ currentTarget: { dataset: { index: newIndex } } });
    }
  },

  togglePlaylist() {
    if (!this.data.showPlaylist) {
      // Load playlist when toggling
      // TODO: Load from server or local storage
    }
    this.setData({ showPlaylist: !this.data.showPlaylist });
  },

  playFromPlaylist(e) {
    const index = e.currentTarget.dataset.index;
    const playlist = this.data.currentPlaylist;
    if (index >= 0 && index < playlist.length) {
      this.setData({ currentMusicIndex: index });
      this.pausePlaying();
      this.loadMusic(playlist[index].id);
      this.startPlaying();
    }
  }
});

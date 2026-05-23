// pages/favorites/favorites.js
const app = getApp();

Page({
  data: {
    favorites: [],
    loading: false
  },

  onLoad() {
    this.loadFavorites();
  },

  onShow() {
    this.loadFavorites();
  },

  loadFavorites() {
    this.setData({ loading: true });

    app.request({
      url: '/favorites',
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          // Fetch music details for each favorite
          const favorites = res.data || [];
          const detailedFavorites = [];

          favorites.forEach(fav => {
            app.request({
              url: `/music/${fav.musicId}`,
              method: 'GET',
              success: (musicRes) => {
                if (musicRes.code === 0) {
                  detailedFavorites.push({
                    ...fav,
                    music: musicRes.data
                  });
                }
              }
            });
          });

          setTimeout(() => {
            this.setData({ favorites: detailedFavorites });
          }, 1000);
        }
      },
      complete: () => {
        this.setData({ loading: false });
      }
    });
  },

  playMusic(e) {
    const musicId = e.currentTarget.dataset.id;
    wx.navigateTo({
      url: `/pages/player/player?id=${musicId}`
    });
  },

  removeFavorite(e) {
    const musicId = e.currentTarget.dataset.id;
    const index = this.data.favorites.findIndex(f => f.musicId === musicId);

    app.request({
      url: `/favorites/${musicId}`,
      method: 'DELETE',
      success: (res) => {
        if (res.code === 0) {
          const favorites = this.data.favorites;
          favorites.splice(index, 1);
          this.setData({ favorites });

          wx.showToast({
            title: 'Removed from favorites',
            icon: 'success',
            duration: 1500
          });
        }
      }
    });
  },

  goBack() {
    wx.navigateBack();
  }
});

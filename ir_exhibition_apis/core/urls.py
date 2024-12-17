from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from search_engine.views import DocumentViewSet
from marketplace.views import ProductViewSet
from timeline.views import HistoricalPeriodViewSet
from ranker.views import DocumentRankerViewSet
from enhanced_ranking_app.views import AdvancedSearchViewSet
from belief_network.views import BeliefNetworkViewSet
from gvsm.views import GVSMDocumentViewSet

router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='documents')
router.register(r'ranker', DocumentRankerViewSet, basename='document-ranker')
router.register(r'advanced-search', AdvancedSearchViewSet, basename='advanced-document')
router.register(r'products', ProductViewSet)
router.register(r'historical-periods', HistoricalPeriodViewSet)
router.register(r'gvsm-documents', GVSMDocumentViewSet, basename='gvsm-documents')
router.register(r'belief-network', BeliefNetworkViewSet, basename='belief-network')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
]
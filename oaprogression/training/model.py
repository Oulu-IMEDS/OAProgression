import pretrainedmodels
from torch import nn



class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SeResNet(nn.Module):
    def __init__(self, layers, drop, ncls):
        super().__init__()
        if layers == 50:
            model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')

        if layers == 101:
            model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')

        if layers == 152:
            model = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')

        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.classifier = nn.Sequential(FCViewer(),
                                            nn.Dropout(drop),
                                            nn.Linear(model.last_linear.in_features, ncls))
        else:
            self.classifier = nn.Sequential(
                FCViewer(),
                nn.Linear(model.last_linear.in_features, ncls)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class KneeNet(nn.Module):
    """
    Network to automatically grade osteoarthritis

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """

    def __init__(self, backbone_net, drop):
        super(KneeNet, self).__init__()
        if backbone_net == 'seresnet50':
            backbone = SeResNet(50, 1, 1)
        else:
            raise NotImplementedError

        self.features = backbone.encoder

        self.classifier_kl = nn.Sequential(nn.Dropout(p=drop),
                                           nn.Linear(backbone.classifier[-1].in_features, 5))

        self.classifier_prog = nn.Sequential(nn.Dropout(p=drop),
                                             nn.Linear(backbone.classifier[-1].in_features, 3))

    def forward(self, x):
        o = self.features(x)
        feats = o.view(o.size(0), -1)
        return self.classifier_kl(feats), self.classifier_prog(feats)
